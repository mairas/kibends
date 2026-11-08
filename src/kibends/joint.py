"""Joint optimisation: every chain and coupled section of a layer as one
L-BFGS-B problem.

The energy is the alternating loop's run energy, the sum of each chain's
energy in its static field plus the other chains on its layer, taken as one
function of every free vertex of the layer: where a quadrature point of one
chain is nearest another chain's edge, the obstacle term's gradient also
reaches that edge's two vertices. Without moving vias the layers do not
interact, so each layer is a problem of its own, and an outer iteration
gives every problem that has not settled one burst: resample, bubbles,
bounded L-BFGS-B, edge check, as in `kibends.optimize`.

Chains of one net on a layer see each other as `neighbours.neighbour_field`
shows them: inflated by their half widths alone, as they may touch, and
left out near where their copper touches at fixed points
(`neighbours.kept_spans`). A selection that branches is split at each
junction into chains that end there, loose (`junctions`): those ends are
one shared variable, with no bending term, as bending is counted at
interior vertices only, and the branches leave each other out near it as
at any touch.

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
given an earlier one's result resumes from it (`optimize_jointly`), which
is how `run` repeats a run with the sections that ended too tight held.

Not yet handled: moving vias.

How much of a same-net chain is left out near a touch depends on the
angles both chains leave it at. A burst fixes it, as pieces of the other
chain's edges, from the shapes the burst starts with, which keeps the
energy smooth within the burst; the next burst measures it again. That is
as often as the loop measures it, once per burst of either chain, and it
errs safe: an angle that closes during the burst meets the other chain's
copper sooner than the start's leave-out expects, which repels and, at
worst, moves edges back; one that opens leaves out more than needed of
copper that is the chain's own net and may touch. The energy that judges
the best state measures the leave-out at the shapes it judges, as the
loop's does.

Why the chains keep their topology
----------------------------------
Static copper is kept as the loop keeps it: each free vertex moves inside a
box inscribed in the disc of half its free distance, so it never jumps an
obstacle, and every edge that newly crosses one after a run moves its free
ends back, with halved bounds for the retry (`bursts.bounded_burst`). A
junction moves inside the smallest of its branches' boxes, and moves back
when any of their edges at it does.

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
at the end could not see, as both chains end clear. Between chains of one
net I is their half widths, still far above a spacing for real tracks; the
parts left out near a touch are not seen at all, as there the chains
connect and may overlap. A junction's branches take no bubble against each
other (`_Problem.siblings`): one net, they connect whichever side of each
other they end up on, so only the edge check keeps them from overlapping.

A section moves as its centreline and is seen as its two tracks, which lie
within H of it, H its half pitch widened at mitred corners, and every point
of the centreline lies within H of each track. A track's vertex can move
farther than the centreline's vertex it is offset from, as the mitre turns
with the edges either side, so the bubbles do not bound the tracks' moves;
the argument needs no such bound, as it is about the centreline. Another
member that ends on the far side of a section without crossing its
centreline lies within H of it, inside the extent I_s the section keeps
clear of it, which is more than H, and the section's own edge check sees it.
So it is enough that no vertex crosses a centreline and no centreline vertex
crosses an edge, with I_s for I above. A chain's vertex p measures its free
distance to the tracks, which it keeps at I_t: at most D + I_s - I_t + H,
less than D + 2I_s, D its free distance to the centreline as the section
keeps it, so p moves at most D/4 + I_s/2 + δ/2. The centreline edge's ends
start within D + I_s + l of p and move at most (D + l)/4 + δ/2 each.
Together that is at most D/2 + I_s/2 + l/4 + δ, less than D + 2I_s for every
D ≥ -I_s whenever l < 4(I_s - δ). The other way round, a centreline vertex
at free distance D from a chain moves at most D/4 + δ/2, and the ends of the
chain's edge, within D + I_s + l of it, have free distances to the tracks of
at most D + l + 2I_s and move at most (D + l)/4 + I_s/2 + δ/2 each: the same
bound. Between two sections each measures the other's tracks, and the same
holds with H on both sides.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

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
from kibends.obstacles import ChainField
from kibends.optimize import (
    MAX_ABNORMAL_BURSTS,
    NOTHING_MOVED,
    PROGRESS_INTERVAL,
    ChainResult,
    FrozenVertex,
    PairResult,
    PolylineStats,
    RunResult,
    SelfCheck,
    Termination,
    frozen_vertices,
    measure,
    resample,
    shape_change,
    stalled,
    violations,
)
from kibends.pair_blocks import PairBlock, PairInput, pair_block

# The share of its free distance to the other chains, which move in the same
# burst, that a vertex's bubble takes: at 1/2 two vertices closing on each
# other cover at most half the distance between them.
CROSS_SHARE = 0.5

End = tuple[int, int]
"""A chain end: the chain's index and 0 or -1."""

Member = ChainInput | PairBlock
"""What a problem optimises: a chain, or a coupled section as its
centreline."""

Resumed = tuple[FloatArray, BoolArray]
"""Where a member resumes: its polyline and the vertices that stay."""


def optimize_jointly(
    chains: Sequence[ChainInput],
    params: EnergyParams,
    settings: OptimizerSettings = DEFAULT_SETTINGS,
    *,
    pairs: Sequence[PairInput] = (),
    clock: Callable[[], float] = time.monotonic,
    started: float | None = None,
    start: RunResult | None = None,
) -> RunResult:
    """Minimise the chains' and sections' summed energy, each layer's jointly.

    Takes and returns what `optimize.optimize_chains` does for plain chains,
    chains whose loose ends meet at junctions (`junctions`), and coupled
    sections of pairs (`pairs`); a junction's ends come back at one point,
    loose. Each problem keeps its best state, replaced only after a burst
    that does not raise its energy and leaves no chain or section with more
    violations than it started with.

    A problem settles when its burst moved no shape by `settled_change` and
    moved no edge back but for a section's parts that came too close to each
    other, or when its best energy improved by less than `stall_tolerance`
    over `stall_window` bursts, or after MAX_ABNORMAL_BURSTS abnormal bursts
    in a row; the run converges when every problem has. With a time limit
    in `settings`, the run stops after the first burst that ends when
    `clock` reads the limit or more past `started`, by default when the run
    starts.

    A run given the result of an earlier one on the same inputs as `start`
    starts each problem from that run's best state, the sections held in
    this run excepted, unless that leaves some chain or section of the
    problem with more violations than its input; then that problem starts
    from its inputs. Either way the run completes its first outer iteration
    before it looks at the time limit.

    Each outer iteration logs its progress at DEBUG, and at INFO once
    `PROGRESS_INTERVAL` has passed since `started` or the last INFO line.
    """
    junctions(chains)
    blocks = [pair_block(pair, settings.spacing) for pair in pairs]
    layers = list(
        dict.fromkeys([*(c.layer for c in chains), *(b.given.layer for b in blocks)])
    )
    indices = [
        [i for i, c in enumerate(chains) if c.layer == layer] for layer in layers
    ]
    sections = [
        [k for k, b in enumerate(blocks) if b.given.layer == layer] for layer in layers
    ]
    problems = [
        _Problem(
            layer,
            [chains[i] for i in members],
            [blocks[k] for k in on_layer],
            params,
            settings,
            None if start is None else _resumed(start, members, on_layer, blocks),
        )
        for layer, members, on_layer in zip(layers, indices, sections, strict=True)
    ]
    moving = [p for p in problems if p.moving]

    def result(termination: Termination, iterations: int, *, best: bool) -> RunResult:
        by_index = {
            i: problem.result(k, best=best)
            for problem, members in zip(problems, indices, strict=True)
            for k, i in enumerate(members)
        }
        by_section = {
            i: problem.pair_result(len(members) + k, best=best)
            for problem, members, on_layer in zip(
                problems, indices, sections, strict=True
            )
            for k, i in enumerate(on_layer)
        }
        return RunResult(
            tuple(by_index[i] for i in range(len(chains))),
            termination,
            iterations,
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
    reach = obstacle_reach(params)
    integrand = obstacle_integrand(params)
    k = params.points_per_edge
    if seen is None:
        seen = seen_parts(chains, polylines, reach)
    lines = [
        shown(member, points) for member, points in zip(chains, polylines, strict=True)
    ]
    flat = np.concatenate([points for parts in lines for _, points in parts])
    at_shown = np.zeros_like(flat)
    gradients: list[FloatArray] = []
    value = 0.0
    for i, (member, points) in enumerate(zip(chains, polylines, strict=True)):
        queries = quadrature_points(points, k)
        static, static_direction = _body(member).field.signed_distance(queries, reach)
        others = _others(seen[i], flat)
        near = others.field.nearest(queries, reach)
        nearer = near.distance < static
        distance = np.where(nearer, near.distance, static)
        direction = np.where(nearer[:, None], near.gradient, static_direction)
        obstacle, obstacle_gradient, slopes = edge_integral(
            points, distance, direction, integrand, k
        )
        shape, shape_gradient = shape_energy(points, masks[i], params, short_edge)
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
        chains, polylines, gradients, lines, strict=True
    ):
        pieces: list[FloatArray] = []
        for _, part in parts:
            pieces.append(at_shown[first : first + len(part)])
            first += len(part)
        if isinstance(member, PairBlock):
            gradient += member.offset_pullback(points, *pieces)
        else:
            gradient += pieces[0]
    if not (math.isfinite(value) and all(np.all(np.isfinite(g)) for g in gradients)):
        raise ValueError(f"energy is not finite: {value}")
    for g, mask in zip(gradients, masks, strict=True):
        g[mask] = 0.0
    return value, gradients


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
    members: Sequence[Member], polylines: Sequence[FloatArray]
) -> FloatArray:
    """What the others see of every member, polylines one after another."""
    return np.concatenate(
        [
            points
            for member, line in zip(members, polylines, strict=True)
            for _, points in shown(member, line)
        ]
    )


@dataclass(frozen=True)
class SeenParts:
    """What one chain of a layer sees of the others: pieces of their edges,
    each inflated by its own amount."""

    vertex: IntArray
    """Each piece's edge's first vertex, numbered across what the others
    see of every chain and section of the layer (`shown`), one after
    another."""
    start: FloatArray
    """Where along its edge each piece starts, from 0 to 1."""
    end: FloatArray
    inflation: FloatArray


def seen_parts(
    chains: Sequence[Member], polylines: Sequence[FloatArray], reach: float
) -> list[SeenParts]:
    """For each chain or section at its polyline, the others at theirs as
    `neighbours.neighbour_field` shows them to it: another net's chain
    whole, inflated by both half widths and the larger clearance, and a
    chain of its net inflated by the half widths alone and left out near
    where they touch. A section sees as its centreline of the pair's outer
    extent, under either of its nets (`PairBlock.as_net`), and is seen as
    its two tracks."""
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
                if j == i:
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
        seen.append(
            SeenParts(
                np.concatenate(vertices),
                np.concatenate(starts),
                np.concatenate(ends),
                np.concatenate(inflations),
            )
        )
    return seen


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
    points: FloatArray, mask: BoolArray, loose: Sequence[int], spacing: float
) -> tuple[FloatArray, BoolArray]:
    """`optimize.resample`, keeping the `loose` ends as loose ends."""
    held = mask.copy()
    held[list(loose)] = True
    points, held = resample(points, held, spacing)
    held[list(loose)] = False
    return points, held


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
class _JointBurst:
    polylines: list[FloatArray]
    rejected_edges: list[int]
    """Per member, its edges whose moves were undone, and a section's
    vertex pairs, over every retry."""
    too_close: list[int]
    """Per member, of `rejected_edges`, a section's vertex pairs that came
    too close to each other (`optimize.close_to_itself`)."""
    abnormal: bool


class _Problem:
    """One layer's chains and coupled sections, optimised together: their
    current shapes, the best state so far, and whether the problem has
    settled. The chains come first, then the sections, as their
    centrelines."""

    def __init__(
        self,
        layer: str,
        chains: Sequence[ChainInput],
        sections: Sequence[PairBlock],
        params: EnergyParams,
        settings: OptimizerSettings,
        resume: Sequence[Resumed | None] | None = None,
    ) -> None:
        self.description = f"the chains on {layer}"
        self.groups = junctions(chains)
        count = len(chains) + len(sections)
        self.loose: list[list[int]] = [[] for _ in range(count)]
        # Branches of one junction leave each other out near it, so what
        # each sees of the others there begins about their half widths plus
        # the reach away: as bubbles, that would hold the vertices near the
        # junction, and with them the junction, to a fraction of the reach
        # per burst. They are one net, so a branch passing through another
        # changes no connection, and the edge check still keeps their edges
        # from newly crossing.
        self.siblings: list[list[int]] = [[] for _ in range(count)]
        for group in self.groups:
            for i, end in group:
                self.loose[i].append(end)
                self.siblings[i] += [k for k, _ in group if k != i]
        # The chains as their neighbours see them; their own masks below
        # leave the junction ends loose.
        self.members: list[Member] = [*_pinned(chains, self.groups), *sections]
        self.bodies = [_body(member) for member in self.members]
        self.params = params
        self.settings = settings
        self.reach = obstacle_reach(params)
        self.originals = [body.centreline.points for body in self.bodies]
        start_fields = self.fields(self.originals)
        self.anchors: list[BoolArray] = []
        self.frozen: list[tuple[FrozenVertex, ...]] = []
        for i, (given, body) in enumerate(
            zip([*chains, *(s.chain for s in sections)], self.bodies, strict=True)
        ):
            others: dict[int, ObstacleField | None] = {
                j: self.neighbour(i, [j], self.originals)
                for j in range(count)
                if j != i
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
        self.free = [i for i, mask in enumerate(self.masks) if not mask.all()]
        self.moving = bool(self.free)
        # Resampling can lay a chord across an obstacle corner, and a burst
        # only rejects crossings its own moves create, so the result is
        # limited to no more violations per member than the resampled start.
        self.allowed = self.violations(self.points, self.masks)
        self.moved = False
        if resume is not None:
            self.resume(resume)
        self.energy = self.energy_of(self.points)
        self.start_energy = self.best_energy = self.energy
        self.best = list(self.points)
        self.best_masks = list(self.masks)
        self.best_energies = [self.best_energy]
        self.rejected_edges = [0] * count
        self.too_close = [0] * count
        self.abnormal_bursts = 0
        self.abnormal_streak = 0
        self.settled = not self.moving

    def resume(self, states: Sequence[Resumed | None]) -> None:
        """Take `states` as the current shapes of the members that move and
        have one, unless that leaves some member with more violations than
        `allowed`."""
        points, masks = list(self.points), list(self.masks)
        for i in self.free:
            state = states[i]
            if state is not None:
                points[i], masks[i] = state
        if all(
            count <= allowed
            for count, allowed in zip(
                self.violations(points, masks), self.allowed, strict=True
            )
        ):
            self.points, self.masks = points, masks
            self.moved = True

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

    def fields(self, polylines: Sequence[FloatArray]) -> list[ObstacleField]:
        """Each member's field with the others at `polylines`."""
        everyone = range(len(self.members))
        return [
            chain_field(
                body.field,
                [self.neighbour(i, [j for j in everyone if j != i], polylines)],
            )
            for i, body in enumerate(self.bodies)
        ]

    def violations(
        self, polylines: Sequence[FloatArray], masks: Sequence[BoolArray]
    ) -> list[int]:
        return [
            violations(points, mask, field)
            for points, mask, field in zip(
                polylines, masks, self.fields(polylines), strict=True
            )
        ]

    def energy_of(self, polylines: Sequence[FloatArray]) -> float:
        return layer_energy(
            self.members, polylines, self.masks, self.params, self.settings.short_edge
        )[0]

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
            abnormal_bursts=0,
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
            abnormal_bursts=0,
            frozen=self.frozen[k],
            before=self.before[k],
            bend=block.bend(points) if moved else block.tight_start,
        )

    def step(self) -> float:
        """One burst of every member together; returns its largest shape
        change."""
        settings = self.settings
        starts: list[FloatArray] = []
        masks: list[BoolArray] = []
        for i, (points, mask) in enumerate(zip(self.points, self.masks, strict=True)):
            if i in self.free:
                points, mask = _resample(points, mask, self.loose[i], settings.spacing)
            starts.append(points)
            masks.append(mask)
        self.masks = masks
        burst = self.burst(starts)
        change = max(
            shape_change(self.points[i], burst.polylines[i]) for i in self.free
        )
        self.moved |= change > NOTHING_MOVED
        self.points = burst.polylines
        for i, (count, close) in enumerate(
            zip(burst.rejected_edges, burst.too_close, strict=True)
        ):
            self.rejected_edges[i] += count
            self.too_close[i] += close
        self.energy = self.energy_of(self.points)
        if self.energy <= self.best_energy and all(
            count <= allowed
            for count, allowed in zip(
                self.violations(self.points, self.masks), self.allowed, strict=True
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
        # of them, not because the shapes settled; but a section pressed
        # onto its own parts beyond a turn has moves to undo in every burst,
        # which no energy term pushes back.
        cut_short = any(
            count > close
            for count, close in zip(burst.rejected_edges, burst.too_close, strict=True)
        )
        self.settled = self.abnormal_streak >= MAX_ABNORMAL_BURSTS or (
            not burst.abnormal
            and (
                (not cut_short and change < settings.settled_change)
                or stalled(self.best_energies, settings)
            )
        )
        return change

    def burst(self, starts: list[FloatArray]) -> _JointBurst:
        """One bounded L-BFGS-B run over every free vertex of the layer,
        retried with halved bounds where edges newly crossed static copper
        or another member's new shape, or a section newly bent below its
        limit or brought its parts beyond a turn too close
        (`optimize.SelfCheck`)."""
        masks = self.masks
        members = self.members
        index, count = _variables(masks, self.groups)
        free = [i >= 0 for i in index]
        seen = seen_parts(members, starts, self.reach)

        def unflatten(x: FloatArray) -> list[FloatArray]:
            at = x.reshape(-1, 2)
            polylines: list[FloatArray] = []
            for start, f, i in zip(starts, free, index, strict=True):
                points = start.copy()
                points[f] = at[i[f]]
                polylines.append(points)
            return polylines

        def objective(x: FloatArray) -> tuple[float, FloatArray]:
            value, gradients = layer_energy(
                members,
                unflatten(x),
                masks,
                self.params,
                self.settings.short_edge,
                seen,
            )
            gradient = np.zeros((count, 2))
            for g, f, i in zip(gradients, free, index, strict=True):
                np.add.at(gradient, i[f], g[f])
            return value, gradient.ravel()

        x0 = np.zeros((count, 2))
        for start, f, i in zip(starts, free, index, strict=True):
            x0[i[f]] = start[f]
        x0 = x0.ravel()
        flat_start = _shown_flat(members, starts)
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

        def crossings(x: FloatArray) -> tuple[list[tuple[int, int]], BoolArray]:
            """Per member, how many of its edges newly cross static copper
            or another member's shape at `x`, and for a section how many
            fail its own check, and of those its vertex pairs too close;
            and the coordinates of the free vertices to move back."""
            polylines = unflatten(x)
            flat = _shown_flat(members, polylines)
            back = np.zeros(count, dtype=bool)
            counts: list[tuple[int, int]] = []
            for i, points in enumerate(polylines):
                a, b = points[:-1], points[1:]
                others = _others(seen[i], flat).field
                crossing = (
                    static_before[i] & ~self.bodies[i].field.segments_clear(a, b)
                ) | (others_before[i] & ~others.segments_clear(a, b))
                ends = np.zeros(len(points), dtype=bool)
                ends[:-1] |= crossing
                ends[1:] |= crossing
                rejected, close = int(crossing.sum()), 0
                check = checks[i]
                if check is not None:
                    own, more, close = check(points)
                    ends |= own
                    rejected += more
                counts.append((rejected, close))
                back[index[i][ends & free[i]]] = True
            return counts, np.repeat(back, 2)

        rounds: list[list[tuple[int, int]]] = []

        def revert(x: FloatArray) -> tuple[BoolArray, int]:
            found: list[list[tuple[int, int]]] = []

            def check(candidate: FloatArray) -> tuple[int, BoolArray]:
                counts, back = crossings(candidate)
                if not found:
                    found.append(counts)
                return sum(rejected for rejected, _ in counts), back

            reverted, total = revert_until_clear(x, x0, check)
            rounds.append(found[0])
            return reverted, total

        sides = np.repeat(self.half_sides(starts, index, seen, count), 2)
        result = bounded_burst(objective, x0, sides, revert, self.settings)
        # An abnormally ended burst stops at its last check without counting it.
        counted = rounds[:-1] if result.abnormal else rounds
        return _JointBurst(
            unflatten(result.points),
            [sum(r[i][0] for r in counted) for i in range(len(starts))],
            [sum(r[i][1] for r in counted) for i in range(len(starts))],
            result.abnormal,
        )

    def half_sides(
        self,
        starts: Sequence[FloatArray],
        index: Sequence[IntArray],
        seen: Sequence[SeenParts],
        count: int,
    ) -> FloatArray:
        """Per variable, the half side of its box: that of the smallest, over
        the vertices it moves, of their free distance to static copper and
        CROSS_SHARE of their free distance to the other members as each
        sees them, less the other branches of its chain's junctions
        (`_Problem.siblings`)."""
        flat = _shown_flat(self.members, starts)
        owner = np.repeat(
            np.arange(len(starts)),
            [
                sum(len(points) for _, points in shown(member, line))
                for member, line in zip(self.members, starts, strict=True)
            ],
        )
        distance = np.full(count, np.inf)
        for i, (body, points) in enumerate(zip(self.bodies, starts, strict=True)):
            free = index[i] >= 0
            static, _ = body.field.signed_distance(points[free])
            apart = ~np.isin(owner[seen[i].vertex], self.siblings[i])
            others = _others(_subset(seen[i], apart), flat).field
            cross, _ = others.signed_distance(points[free])
            np.minimum.at(
                distance, index[i][free], np.minimum(static, CROSS_SHARE * cross)
            )
        return half_side(distance, self.settings)
