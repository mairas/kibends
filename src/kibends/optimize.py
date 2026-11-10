"""The optimizer: every chain, coupled section and moving via of a problem
as one L-BFGS-B problem.

This module holds the run loop (`optimize_chains`), which splits the
members into problems and gives them bursts until they settle, and
`_Problem`, one problem's variables, energy, checks and best state. What
does not depend on a problem lives beside it: `members` (what the members
see of each other and their summed energy), `polylines` (resampling,
statistics, violations, shape change and a section's self check),
`contacts` (the stops that keep a vertex on its contact) and `results`
(what a run returns).

The energy of a problem is the sum of each chain's energy in its static
field plus the other chains on its layer (`members.members_energy`), taken
as one function of every free vertex: where a quadrature point of one chain
is nearest another chain's edge, the obstacle term's gradient also reaches
that edge's two vertices. Without moving vias the layers do not interact,
so each layer is a problem of its own, and an outer iteration gives every
problem that has not settled one burst: resample each chain to the spacing
between its fixed vertices and those pressed into the others' clearance,
which resampling keeps where they are (`_Problem.pressed`), run L-BFGS-B
(`bursts.run_burst`) over every free vertex without bounds, and move back
what the checks reject.

Copper of one net does not interact: a chain sees nothing of the chains,
tracks and vias of its net (`members.seen_parts`), and its static
field holds none of its net's copper, so chains of one net may come close,
overlap or cross. A selection that branches is split at each junction into
chains that end there, loose (`junctions`): those ends are one shared
variable, and the branches, of one net, do not see each other.

A coupled section of a differential pair (`kibends.pair_blocks`) is a
member of its layer's problem as its centreline: its free vertices are
variables, its energy has the bend-radius hinge, and it keeps clear of the
others as the pair's outer extent, its own nets' chains included but for
near its fan-outs (`PairBlock.as_net`), since its other track must keep
clear of them. The others see its two tracks, a chain of one of the pair's
nets only the track of the other net, offset from the centreline at each
evaluation (`members.shown`), so where the obstacle term pulls on a
track's edge, the gradient goes back through the offset to the
centreline's vertices (`PairBlock.offset_pullback`). Besides the edge
check, a burst moves back what `polylines.SelfCheck` finds: a centreline
newly bent below its limit, or brought closer to its own parts beyond a
turn than it may. A held section is fully pinned and not resampled; a run
given an earlier one's result resumes from it (`optimize_chains`), which
is how `run` repeats a run with the sections that ended too tight held.

A moving via (`kibends.via_moves`) is two more variables, its centre. The
ends of the chains it joins are not variables of their own: each is the
via's centre plus its offset at the start, at every evaluation, so they keep
their offsets exactly and their gradient goes to the via. Its own obstacle
terms are its ring's, standing for a track as long as it is wide, against
its static field over every layer it spans, the members of other nets near
it on those layers and the vias of other nets it does not pair with (below),
each the nearest of them, with their gradient also at
the edge or via that answers it, and the clearance hinge alone, which limits
without repelling, against the outside of its zones' regions and the
fabrication rules that hold within its net: hole-to-hole from its net's
vias, moving or standing still, and plated pads, and no overlap with its
net's pads (`obstacles.own_net_stops`). The members of other nets see it as
a disc on each of its layers, so that where the obstacle term pulls on it,
the gradient reaches its centre too. A section and a via see each other as the section's
centreline as wide as `PairBlock.chain`. A stitching via carries no chain
ends, so with no length term only repulsion moves it. A moving via ties the
layers it spans into one problem, every member on them included, since moving vias are
through vias and share every layer.

The board's vias that are none of the run's, each a ground via or a
regular one (`vias.StaticVia`), reach `optimize_chains` as `static_vias`;
a problem keeps those its moving vias share a layer with
(`_Problem.static_vias`). A ground via and a regular one, ground vias of
two ground nets, and ground vias of one net interact through the pairwise
term (`via_pairs`), summed over every such pair, moving or standing still,
rather than the nearest only: the first attract each other and keep clear
of each other, the second keep clear, both at the distances the via's
limits give, and the third repel each other beyond their rings and
clearance while their stops keep hole-to-hole. Without such a pair the
problem's energy is the one without ground vias, term for term.

How much of a chain of one of its nets a section leaves out near a fan-out
depends on the angles both leave it at. A burst fixes it, as pieces of the
chain's edges, from the shapes the burst starts with, which keeps the
energy smooth within the burst; the next burst measures it again. That
errs safe: an angle that closes during the burst meets the chain's copper
sooner than the start's leave-out expects, which repels and, at worst,
moves edges back; one that opens leaves out more than needed of copper
that keeps clear of the section's other track by itself. The energy that
judges the best state measures the leave-out at the shapes it judges.

What a burst checks
-------------------
A burst runs L-BFGS-B over every free vertex and moving via of the problem
without bounds (`bursts.run_burst`), and the checks after it compare where
the burst ends with where it started. An edge that starts clear of static
copper, or of the other members as it sees them, and ends crossing it
moves its free ends back to the burst's start, and the check repeats, as
moving one member's vertex back can put another member's moved edge
across it; moved back whole, every member is at its start, clear
(`bursts.revert_until_clear`). A junction moves back when any of its
branches' edges at it does. An edge that starts inside another member's
clearance is not checked; the violation guard on the best state counts
it.

A section moves as its centreline and is seen as its two tracks, which
the others' edge checks see where the burst leaves them; its own edges
are checked as the pair's outer extent. A burst also moves back what
`polylines.SelfCheck` finds on a section: an edge newly bent below its
limit, or its parts beyond a turn brought closer than they may
(`polylines.close_to_itself`).

A moving via is a member whose one vertex is its centre, inflated by
`via_chain_inflation` against a member of another net and seen by it so,
and against another via by `via_via_inflation`, hole-to-hole alone for one
of its net. The chain ends it carries move with it, and their edges are
checked as any. A via clear at the burst's start of the copper of other
nets, static (`ViaInput.copper`), a member or a via, that ends inside its
clearance of it moves back with the ends it carries, and so does any via
that ends deeper inside what only stops it than it started the burst and
inside it: outside where its zones, taken together on each layer where one
reaches the via's ring, connect it at the next refill
(`obstacles.containment`), or inside its own net's stops (`ViaInput.stops`,
its net's other vias). The two are measured apart, so that a via inside
one cannot be pushed deeper into the other. On those
layers its inflation also holds the zone's clearance and margin, against
static copper and members alike, so that the refill reaches the via
wherever the others move. The chains a via carries are of its net, so it
does not see them, nor they it.

A vertex that keeps a pad or a via that stays (`ChainInput.contacts`)
is free: it has a stop, the clearance hinge at its distance to the
region's edge (`contacts.contact_stops`), which limits without
repelling, and the check below moves it back if it newly ends outside
(`contacts.contacts_left`).
Every state a burst leaves, and so the best state, has it inside.

A vertex or via that starts inside its clearance moves like any other: the
clearance hinge pushes it out, and the check after a burst moves back only
what was clear at the burst's start. Resampling keeps a vertex more than
PRESSED_DEPTH inside the others' clearance where it is, as a free vertex
(`_Problem.pressed`). Each problem keeps its best
state, taken only from a burst that does not raise its energy, add
violations to any member or leave a via deeper inside its clearance or
its stops than it started.

When it stops
-------------
A problem settles when its burst moves nothing by `settled_change` and
moves nothing back but a section's own parts, when its lowest energy
stalls over `stall_window` bursts, or after MAX_ABNORMAL_BURSTS abnormal
bursts in a row. The run converges when every problem has settled, none
of them on abnormal bursts, and otherwise stops at the iteration cap or
after the first burst that ends past the time limit, counted from the
start of the whole stage, keeping each problem's best state either way
(`optimize_chains`). A run in which some problem settled on abnormal
bursts gave up: it keeps the best state if that is lower in energy than
the inputs, from which a resumed run measures too, and ends abnormally if
not.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from loguru import logger

from kibends.bursts import (
    DEFAULT_SETTINGS,
    INSIDE_TOLERANCE,
    OptimizerSettings,
    point_distance,
    revert_until_clear,
    run_burst,
)
from kibends.contacts import contact_stops, contacts_left, no_contacts
from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    IntArray,
    Integrand,
    ObstacleField,
    hinge_integrand,
    obstacle_integrand,
    obstacle_reach,
)
from kibends.members import (
    Member,
    body_of,
    members_energy,
    pull_answers,
    seen_parts,
    shown,
    shown_flat,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE, ChainCentreline
from kibends.pair_blocks import PairBlock, PairInput, pair_block
from kibends.polylines import (
    SelfCheck,
    clearance_points,
    count_violations,
    measure,
    measured,
    resample,
    shape_change,
)
from kibends.results import (
    ChainResult,
    OptimizeResult,
    PairResult,
    PolylineStats,
    RunResult,
    Termination,
    ViaResult,
)
from kibends.seen import NeighbourList, Seen, ViaSight
from kibends.via_moves import MovingVias, free_distances
from kibends.via_pairs import ViaPairs
from kibends.vias import StaticVia, ViaInput

# After this many abnormal L-BFGS-B ends in a row a problem counts as
# settled and keeps what its bursts reached.
MAX_ABNORMAL_BURSTS = 3

# A run whose steps change no shape by more than this, KiCad's 1 nm grid,
# mm, has nothing to do, however its stop rules are set.
NOTHING_MOVED = 1e-6

# A run logs its progress at INFO at most this often, seconds, and not at all
# if it ends sooner; DEBUG gets every outer iteration.
PROGRESS_INTERVAL = 5.0


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
        before=chain.before,
        after=after,
    )


def stalled(lowest_energies: list[float], settings: OptimizerSettings) -> bool:
    if len(lowest_energies) <= settings.stall_window:
        return False
    earlier = lowest_energies[-1 - settings.stall_window]
    return earlier - lowest_energies[-1] <= settings.stall_tolerance


# How far, mm, the points of a member and the pieces it sees may move before
# it finds its neighbours again (`NeighbourList`): a wider skin lists more
# pairs at every evaluation, a narrower one searches more often.
LIST_SKIN = 0.05

# How deep inside the other members' and vias' clearance, mm, a vertex lies
# when resampling keeps it where it is (`_Problem.pressed`), so that one that
# only grazes a neighbour's clearance resamples as any other.
PRESSED_DEPTH = 0.02

End = tuple[int, int]
"""A chain end: the chain's index and 0 or -1."""

Resumed = tuple[FloatArray, BoolArray, IntArray]
"""Where a member resumes: its polyline, the vertices that stay and, per
vertex, the contact it keeps (`_Problem.contact_of`)."""


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
    static_vias: Sequence[StaticVia] = (),
) -> RunResult:
    """Minimise the chains', sections' and vias' summed energy, each
    problem's jointly: the layers that moving vias tie together are one
    problem, and every other layer is one of its own.

    Takes chains whose ends are fixed or loose where they meet at junctions
    (`junctions`), coupled sections of pairs (`pairs`) and moving `vias`,
    and returns each chain's, section's and via's best state; a junction's
    ends come back at one point, loose. A via keeps `hole_clearance` and
    `hole_to_hole` where holes are involved. `static_vias` are the board's
    vias that are none of `vias`; each problem keeps those its moving vias
    share a layer with. Each problem keeps its best state, replaced only after a
    burst that does not raise its energy, leaves no chain or section with
    more violations than it started with, and leaves no via deeper inside
    its clearance or its stops than it started.

    A problem settles when its burst moved no shape or via by
    `settled_change` and moved nothing back but for a section's parts that
    came too close to each other, or when the lowest energy its bursts
    reached fell by less than `stall_tolerance` over `stall_window` bursts,
    whether or not the guard took them as its best state, or after
    MAX_ABNORMAL_BURSTS abnormal bursts in a row; the run converges when
    every problem has, and gives up (`Termination.GAVE_UP`) if some problem
    settled on abnormal bursts. A run that gave up on some problem and
    whose best states sum to no less energy than its inputs, not where it
    resumed, ends `Termination.ABNORMAL`, whatever stopped it. With a time
    limit in `settings`, the run stops after the first burst that ends when
    `clock` reads the limit or more past `started`, by default when the run
    starts.

    A run given the result of an earlier one on the same inputs as `start`
    starts each problem from that run's best state, the sections held in
    this run excepted, unless that leaves some chain or section of the
    problem with more violations or more edges across the others than its
    input, or a via newly inside its clearance; then that problem starts
    from its inputs. Either way the run
    completes its first outer iteration before it looks at the time limit.

    It logs at INFO when setup begins and, once the problems are built,
    the setup time and the starting length and energy, saying "again" when
    it repeats a run from `start` with sections held. Each outer iteration
    logs its progress at DEBUG, and at INFO once `PROGRESS_INTERVAL` has
    passed since `started` or the last INFO line. Each burst logs what it
    cost at DEBUG only (`_Problem.step`).
    """
    again = start is not None
    logger.info(
        "Setting up the optimizer again, with held sections"
        if again
        else "Setting up the optimizer"
    )
    set_up = clock()
    junctions(chains)
    blocks = [pair_block(pair, settings.spacing) for pair in pairs]
    everything = [*chains, *(block.chain for block in blocks)]
    moving_vias = MovingVias(everything, vias, hole_clearance, hole_to_hole)
    placed = _problems(
        chains, blocks, moving_vias, params, settings, start, static_vias
    )
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
        }
        return RunResult(
            tuple(by_index[i] for i in range(len(chains))),
            termination,
            iterations,
            vias=tuple(by_via[v] for v in range(len(vias))),
            pairs=tuple(by_section[k] for k in range(len(blocks))),
            gave_up=tuple(
                (p.description, p.abnormal_streak) for p in problems if p.gave_up
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
    logger.info(_starting(moving, begun - set_up, again=again))
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
            termination = (
                Termination.GAVE_UP
                if any(p.gave_up for p in moving)
                else Termination.CONVERGED
            )
            break
        if out_of_time:
            termination = Termination.TIME_LIMIT
            break
    final = result(termination, iteration, best=True)
    if final.gave_up and sum(p.best_energy for p in moving) >= sum(
        p.input_energy for p in moving
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
    static_vias: Sequence[StaticVia] = (),
) -> list[_Placed]:
    """The run's problems: the layers that the vias tie together, with
    every via and the `static_vias` they share a layer with, and each other
    layer alone. `vias` holds the chains, then the sections' centrelines."""
    layers = list(dict.fromkeys(chain.layer for chain in vias.chains))
    tied = [
        layer for layer in layers if any(via.via.reaches(layer) for via in vias.vias)
    ]
    if not tied:
        # Vias on no chain's layer, stitching vias selected alone.
        tied = list(dict.fromkeys(layer for via in vias.vias for layer in via.via.span))
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
            [_problem_via(vias, v, local, joined) for v in joined],
            params,
            settings,
            resume,
            static_vias,
            hole_clearance=vias.hole_clearance,
            hole_to_hole=vias.hole_to_hole,
        )
        placed.append(_Placed(problem, on, sections, joined))
    return placed


def _problem_via(
    vias: MovingVias,
    v: int,
    local: Mapping[int, int],
    joined: Sequence[int],
) -> _Via:
    """Via v of `vias` in a problem whose members `local` numbers and whose
    vias are `joined`."""
    apart: dict[int, float] = {}
    for k, u in enumerate(joined):
        if u == v:
            continue
        inflation = vias.apart(v, u)
        if inflation is not None:
            apart[k] = inflation
    return _Via(
        vias.vias[v],
        tuple((local[e.chain], e.index, e.offset) for e in vias.ends[v]),
        {local[i]: ring for i, ring in vias.rings(v).items() if i in local},
        apart,
    )


@dataclass(frozen=True)
class _Via:
    """A via of a problem, with the problem's members and vias by their
    index in it."""

    given: ViaInput
    carried: tuple[tuple[int, int, FloatArray], ...]
    """The chain ends it carries: the member, 0 or -1 for its first or last
    point, and the end's offset from the via's centre."""
    rings: Mapping[int, float]
    """The members it sees and that see it, and how far their centrelines
    keep from its centre."""
    apart: Mapping[int, float]
    """The other vias it keeps clear of, and how far apart their centres
    keep."""


@dataclass(frozen=True)
class _ViaPart:
    """One of each moving via's two terms: the integrand applied to the
    nearest of what `static` gives of the via and what `sight` finds."""

    static: Callable[[ViaInput], ObstacleField | None]
    sight: ViaSight
    integrand: Integrand


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
    resumed: list[Resumed | None] = []
    for i in chains:
        chain = start.chains[i]
        contact_of = chain.contact_of
        resumed.append((chain.polyline, chain.fixed & (contact_of < 0), contact_of))
    for k in sections:
        block = blocks[k]
        if block.held:
            resumed.append(None)
            continue
        points = start.pairs[k].centreline
        pinned = start.pairs[k].fixed | block.direction_pins(points)
        resumed.append((points, pinned, no_contacts(points)))
    return resumed


def junctions(chains: Sequence[ChainInput]) -> list[list[End]]:
    """The chain ends that are neither fixed nor keep a contact, grouped by
    the point, layer and net they share; raises ValueError for one that
    meets no other."""
    groups: dict[tuple[str, str, float, float], list[End]] = {}
    for i, chain in enumerate(chains):
        sliding = {v for v, _ in chain.contacts}
        for end, at in ((0, 0), (-1, len(chain.fixed) - 1)):
            if not chain.fixed[end] and at not in sliding:
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


def _starting(problems: Sequence[_Problem], setup: float, *, again: bool) -> str:
    """One line on the run about to start, after `setup` seconds of setup,
    `again` when it repeats a run with sections held."""
    count = len(problems)
    return (
        f"Optimizer set up{' again' if again else ''} in {setup:.1f} s: "
        f"{count} problem{'' if count == 1 else 's'}, "
        f"length {sum(p.length() for p in problems):.3f} mm, "
        f"energy {sum(p.energy for p in problems):.3f}"
    )


def _progress(
    problems: Sequence[_Problem], iteration: int, elapsed: float, change: float
) -> str:
    """One line on where the run stands after an outer iteration."""
    return (
        f"Iteration {iteration}, {elapsed:.1f} s: "
        f"length {sum(p.length() for p in problems):.3f} mm, "
        f"energy {sum(p.energy for p in problems):.3f}, "
        f"largest change {change:.2e} mm"
    )


def _resample(
    points: FloatArray,
    mask: BoolArray,
    contact_of: IntArray,
    loose: Sequence[int],
    spacing: float,
    kept: BoolArray | None = None,
) -> tuple[FloatArray, BoolArray, IntArray]:
    """`polylines.resample`, keeping the `loose` ends, the vertices that keep
    a contact (`contact_of`) and the `kept` vertices where they are, as free
    vertices, and where each contact went."""
    anchors = mask | (contact_of >= 0)
    anchors[list(loose)] = True
    if kept is not None:
        anchors |= kept
    points, fixed = resample(points, anchors, spacing)
    # Resampling keeps the anchors, in order.
    at = np.flatnonzero(fixed)
    fixed[at[~mask[anchors]]] = False
    moved = no_contacts(points)
    moved[at] = contact_of[anchors]
    return points, fixed, moved


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
    centres: each via's centre is a variable, and an end it carries is that
    variable plus the end's offset."""

    starts: list[FloatArray]
    positions: list[FloatArray]
    index: list[IntArray]
    """Per member, each vertex's variable, -1 where it stays."""
    offsets: list[FloatArray]
    """Per member, each vertex's offset from its variable."""
    via_index: IntArray
    """Each via's variable."""
    count: int

    @property
    def x0(self) -> FloatArray:
        x = np.zeros((self.count, 2))
        for start, i, offset in zip(self.starts, self.index, self.offsets, strict=True):
            free = i >= 0
            x[i[free]] = start[free] - offset[free]
        for v, at in zip(self.via_index, self.positions, strict=True):
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
        positions = [at[v].copy() for v in self.via_index]
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
        np.add.at(gradient, self.via_index, via_gradients)
        return gradient.ravel()


@dataclass(frozen=True)
class _Round:
    """What one check of a burst found: per member, its edges that newly
    crossed and, of those, a section's vertex pairs too close; per via, what
    moved it back."""

    members: list[tuple[int, int]]
    vias: list[int]


@dataclass(frozen=True)
class _ProblemBurst:
    polylines: list[FloatArray]
    positions: list[FloatArray]
    rejected_edges: list[int]
    """Per member, its edges whose moves were undone, and a section's
    vertex pairs."""
    too_close: list[int]
    """Per member, of `rejected_edges`, a section's vertex pairs that came
    too close to each other (`polylines.close_to_itself`)."""
    via_rejected: list[int]
    """Per via, the edges at the ends it carries that newly crossed, and
    the times it ended newly inside its clearance."""
    abnormal: bool
    iterations: int = 0
    """The L-BFGS-B iterations it ran."""
    evaluations: int = 0
    """The times L-BFGS-B evaluated the energy."""
    message: str = ""
    """What L-BFGS-B says ended it; empty for a burst that did not run it."""
    evaluating: float = 0.0
    """Seconds those evaluations took."""


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
        static_vias: Sequence[StaticVia] = (),
        *,
        hole_clearance: float,
        hole_to_hole: float,
    ) -> None:
        self.vias = list(vias)
        self.static_vias = [
            s
            for s in static_vias
            if any(s.via.shares_layer(via.given.via) for via in vias)
        ]
        pairs = ViaPairs(
            [via.given for via in vias],
            [via.apart for via in vias],
            self.static_vias,
            params,
            hole_clearance=hole_clearance,
            hole_to_hole=hole_to_hole,
        )
        self.pairs = None if pairs.empty else pairs
        """The pairwise term (`via_pairs`); None without a pair, so that
        the energy is the one without ground vias term for term."""
        self.via_diameters = np.array(
            [via.given.via.diameter for via in vias], dtype=float
        )
        self.description = _description(layers, bool(vias))
        self.groups = junctions(chains)
        count = len(chains) + len(sections)
        self.loose: list[list[int]] = [[] for _ in range(count)]
        for group in self.groups:
            for i, end in group:
                self.loose[i].append(end)
        self.carried: list[list[tuple[int, int]]] = [[] for _ in range(count)]
        for k, via in enumerate(vias):
            for i, end, _ in via.carried:
                self.carried[i].append((k, end))
        self.rings = [via.rings for via in vias]
        apart = [via.apart for via in vias]
        # Vias of one net only stop each other; the pairs of the pairwise
        # term are summed apart from the nearest; the rest repel.
        own_net = [
            {u: d for u, d in a.items() if _one_net(vias[k], vias[u])}
            for k, a in enumerate(apart)
        ]
        self.via_sight = ViaSight(
            self.rings,
            [
                {u: d for u, d in a.items() if u not in own}
                for a, own in zip(apart, own_net, strict=True)
            ],
            count,
        )
        paired = pairs.paired
        self.via_repel = ViaSight(
            self.rings,
            [
                {u: d for u, d in a.items() if u not in own and u not in pairs_of}
                for a, own, pairs_of in zip(apart, own_net, paired, strict=True)
            ],
            count,
        )
        self.via_stop = ViaSight([{} for _ in vias], own_net, count)
        self.members: list[Member] = [*chains, *sections]
        self.bodies = [body_of(member) for member in self.members]
        self.params = params
        self.settings = settings
        self.reach = obstacle_reach(params)
        self.integrand = obstacle_integrand(params)
        self.limit = hinge_integrand(params)
        self.via_parts = (
            _ViaPart(lambda given: given.field, self.via_repel, self.integrand),
            _ViaPart(lambda given: given.stops, self.via_stop, self.limit),
        )
        self.originals = [body.centreline.points for body in self.bodies]
        self.start_positions = [
            np.array(via.given.via.position, dtype=float) for via in vias
        ]
        self.anchors = [body.fixed for body in self.bodies]
        # A vertex that stays keeps its contact without one.
        self.within = [[field for _, field in body.contacts] for body in self.bodies]
        self.sliding = [i for i, body in enumerate(self.bodies) if body.contacts]
        self.given_contact_of: list[IntArray] = []
        for body, anchors in zip(self.bodies, self.anchors, strict=True):
            contact_of = no_contacts(body.centreline.points)
            for k, (v, _) in enumerate(body.contacts):
                contact_of[v] = k
            contact_of[anchors] = -1
            self.given_contact_of.append(contact_of)
        self.before = self.start_statistics(self.originals, self.start_positions)
        # A burst starts from the members resampled; the shapes as the burst
        # left them are what the energy, the guard and the best state see.
        # Resampling would put free vertices between a held section's pins.
        self.points: list[FloatArray] = []
        self.masks: list[BoolArray] = []
        self.contact_of: list[IntArray] = []
        """Per member, each vertex's contact among `within`, -1 for none."""
        for member, original, anchors, given, loose in zip(
            self.members,
            self.originals,
            self.anchors,
            self.given_contact_of,
            self.loose,
            strict=True,
        ):
            if isinstance(member, PairBlock) and member.held:
                points, mask, contact_of = original, anchors, given
            else:
                points, mask, contact_of = _resample(
                    original, anchors, given, loose, settings.spacing
                )
            moves = not mask.all()
            self.points.append(points if moves else original)
            self.masks.append(mask if moves else anchors)
            self.contact_of.append(contact_of if moves else given)
        self.positions = list(self.start_positions)
        self.free = [
            i for i, mask in enumerate(self.masks) if not mask.all() or self.carried[i]
        ]
        self.moving = bool(self.free) or bool(vias)
        # Resampling can lay a chord across an obstacle corner, and a burst
        # only rejects crossings its own moves create, so the result is
        # limited to no more violations per member than the resampled start,
        # and to no via deeper inside its clearance or its stops than it
        # started.
        self.seen = self.seeing(self.points)
        """What the members see of each other at `points`."""
        self.allowed = self.violations(
            self.points, self.masks, self.positions, self.seen
        )
        self.inside = self.via_distances(self.points, self.positions)
        self.stopped = self.stop_distances(self.points, self.positions)
        self.moved = False
        # The first burst starts from the shapes `allowed` was measured on:
        # resampled again, they would cut corners deeper still.
        self.resampled = [True] * count
        self.energy = self.energy_of(self.points, self.positions, self.seen)
        # A resumed run has improved on its inputs, not only on where it
        # resumes, if its best state beats them.
        self.input_energy = self.energy
        if resume is not None:
            self.resume(resume)
            self.energy = self.energy_of(self.points, self.positions, self.seen)
        self.best_energy = self.energy
        self.best = list(self.points)
        self.best_masks = list(self.masks)
        self.best_contact_of = list(self.contact_of)
        self.best_positions = list(self.positions)
        self.lowest = [self.energy]
        self.rejected_edges = [0] * count
        self.too_close = [0] * count
        self.via_rejected = [0] * len(vias)
        self.abnormal_bursts = 0
        self.abnormal_streak = 0
        self.settled = not self.moving

    @property
    def gave_up(self) -> bool:
        """Whether it settled on MAX_ABNORMAL_BURSTS abnormal bursts in a row."""
        return self.abnormal_streak >= MAX_ABNORMAL_BURSTS

    def resume(self, state: _Resume) -> None:
        """Take `state` as the current shapes of the members that move and
        have one, and the vias' centres, unless that leaves some member with
        more violations than `allowed`, or with more edges across the others
        than at its inputs, or some via deeper inside its clearance than it
        started and inside it. Edges are counted too because a held section
        is all pins, which `polylines.violations` cannot count, and a burst leaves an
        edge that starts across another member as it is."""
        points, masks = list(self.points), list(self.masks)
        contact_of = list(self.contact_of)
        for i in self.free:
            given = state.members[i]
            if given is not None:
                points[i], masks[i], contact_of[i] = given
        positions = list(state.positions)
        seen = self.seeing(points)
        across = zip(
            self.edges_across(points, positions, seen),
            self.edges_across(self.points, self.positions, self.seen),
            strict=True,
        )
        if self.guard(points, masks, positions, seen) and all(
            count <= before for count, before in across
        ):
            self.points, self.masks, self.positions = points, masks, positions
            self.seen = seen
            self.contact_of = contact_of
            self.resampled = [
                done and given is None
                for done, given in zip(self.resampled, state.members, strict=True)
            ]
            self.moved = True

    def seeing(self, polylines: Sequence[FloatArray]) -> Seen:
        """What each member sees of the others and of the vias with the
        members at `polylines` (`members.seen_parts`)."""
        return seen_parts(self.members, polylines, self.reach, self.rings)

    def start_statistics(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> list[PolylineStats]:
        """Each member's statistics at `polylines` (`polylines.measure`),
        its clearance taken from its static copper, the other members and
        the vias at `positions`."""
        spacing = self.settings.spacing
        samples = [clearance_points(points, spacing) for points in polylines]
        flat = shown_flat(self.members, polylines, positions)
        near = self.seeing(polylines).distances(samples, flat)
        return [
            measured(points, np.minimum(body.field.signed_distance(at)[0], d), spacing)
            for body, points, at, d in zip(
                self.bodies, polylines, samples, near, strict=True
            )
        ]

    def edges_across(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> list[int]:
        """Per member at `polylines`, its edges not clear of the other
        members and the vias at `positions` as it sees them, `seen` there."""
        flat = shown_flat(self.members, polylines, positions)
        return [
            int(np.count_nonzero(~clear))
            for clear in seen.segments_clear(polylines, flat)
        ]

    def guard(
        self,
        polylines: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> bool:
        """Whether no member has more violations than `allowed` and no via
        is deeper inside its clearance, or inside its stops, than it started,
        and inside them. The two are measured apart, so that a via inside
        one cannot be pushed deeper into the other. `seen` is what the
        members see at `polylines`."""
        return (
            all(
                count <= allowed
                for count, allowed in zip(
                    self.violations(polylines, masks, positions, seen),
                    self.allowed,
                    strict=True,
                )
            )
            and _no_deeper(self.via_distances(polylines, positions), self.inside)
            and _no_deeper(self.stop_distances(polylines, positions), self.stopped)
        )

    def violations(
        self,
        polylines: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> list[int]:
        """Per member at `polylines`, its `polylines.violations` of its
        static copper, the other members and the vias at `positions`
        together, every member's at once through `seen`, what they see
        there."""
        flat = shown_flat(self.members, polylines, positions)
        near = seen.distances(polylines, flat, -INSIDE_TOLERANCE)
        clear = seen.segments_clear(polylines, flat, INSIDE_TOLERANCE)
        counts: list[int] = []
        for body, points, mask, distance, others_clear in zip(
            self.bodies, polylines, masks, near, clear, strict=True
        ):
            static = body.field
            inside = (static.signed_distance(points)[0] < -INSIDE_TOLERANCE) | (
                distance < -INSIDE_TOLERANCE
            )
            ends_clear = ~inside[:-1] & ~inside[1:]
            cut = ~others_clear
            cut[ends_clear] |= ~static.segments_clear(
                points[:-1][ends_clear], points[1:][ends_clear], INSIDE_TOLERANCE
            )
            counts.append(count_violations(inside, cut, mask))
        return counts

    def via_distances(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> list[float]:
        """Each moving via's free distance, in order, to the copper of other
        nets: its static copper, the members near it at `polylines` and the
        vias of other nets at `positions`. What only stops it is
        `stop_distances`."""
        return free_distances(
            [via.given for via in self.vias], self.via_sight, polylines, positions
        )

    def stop_distances(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> list[float]:
        """Each moving via's distance, in order, to what only stops it: its
        stops (`ViaInput.stops`) and the other vias of its net at
        `positions`."""
        near = self.via_stop.nearest(polylines, positions, math.inf).distance
        return [
            float(d)
            if via.given.stops is None
            else min(point_distance(via.given.stops, at), float(d))
            for via, at, d in zip(self.vias, positions, near, strict=True)
        ]

    def via_energy(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> tuple[float, list[FloatArray], FloatArray]:
        """The moving vias' own obstacle terms, each via's ring standing for
        a track as long as it is wide, and their gradient at every vertex and
        at every via's centre. Where a member's edge or another via answers a
        via, the gradient also pulls on it.

        Each via has two terms, each for the nearest of what it covers: the
        full integrand for its static field and the members and the vias of
        other nets it does not pair with (`via_pairs` has those), and the
        clearance hinge alone for what only stops it, its
        zones' regions and the vias of its net, moving or standing still."""
        gradients = [np.zeros_like(points) for points in polylines]
        via_gradients = np.zeros((len(self.vias), 2))
        if not self.vias:
            return 0.0, gradients, via_gradients
        at = np.array(positions)
        distances, directions = self.via_statics(at)
        value = 0.0
        for part, static, toward in zip(
            self.via_parts, distances, directions, strict=True
        ):
            near = part.sight.nearest(polylines, positions, self.reach)
            nearer = near.distance < static
            distance = np.where(nearer, near.distance, static)
            direction = np.where(nearer[:, None], near.gradient, toward)
            term, slope = part.integrand(distance)
            value += float(self.via_diameters @ term)
            push = (self.via_diameters * slope)[:, None] * direction
            via_gradients += push
            pull_answers(near, nearer, at, push, gradients, via_gradients)
        return value, gradients, via_gradients

    def via_statics(self, at: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The distance (parts, n) and unit gradient (parts, n, 2) of each
        moving via, at its row of `at`, to what the static field of each of
        `via_parts` gives of it, in one loop over the vias; +inf and zero
        where it gives none or it is beyond the reach."""
        distance = np.full((len(self.via_parts), len(at)), math.inf)
        direction = np.zeros((len(self.via_parts), len(at), 2))
        for n, (via, point) in enumerate(zip(self.vias, at, strict=True)):
            given = via.given
            for p, part in enumerate(self.via_parts):
                field = part.static(given)
                if field is not None:
                    d, g = field.signed_distance(point[None], self.reach)
                    distance[p, n], direction[p, n] = d[0], g[0]
        return distance, direction

    def left(self, i: int, points: FloatArray) -> BoolArray:
        """Member i's vertices at `points` that lie outside the contact
        they keep (`contacts.contacts_left`)."""
        if i not in self.sliding:
            return np.zeros(len(points), dtype=bool)
        return contacts_left(points, self.contact_of[i], self.within[i])

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
        seen: Seen | None = None,
        listed: NeighbourList | None = None,
    ) -> tuple[float, list[FloatArray], FloatArray]:
        """The problem's energy, by default with what each member sees of
        the others as `members.seen_parts` finds it at `polylines`, and its gradient
        at every vertex and every via's centre; `listed` as for
        `members.members_energy`."""
        if seen is None:
            seen = self.seeing(polylines)
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
        if self.pairs is not None:
            paired, paired_gradients = self.pairs.energy(positions)
            own += paired
            own_gradients += paired_gradients
        for i in self.sliding:
            points = polylines[i]
            stop, push = contact_stops(
                points,
                self.contact_of[i],
                self.within[i],
                self.bodies[i].centreline.width,
                self.limit,
            )
            value += stop
            gradients[i] += push
        return value + own, gradients, via_gradients + own_gradients

    def energy_of(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> float:
        """The problem's energy, with what the members see of each other at
        `polylines` given as `seen`."""
        return self.total(polylines, positions, self.masks, seen)[0]

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
        contact_of = self.best_contact_of[k] if best else self.given_contact_of[k]
        mask = self.best_masks[k] if best else self.anchors[k]
        return ChainResult(
            polyline=self.best[k] if best else self.originals[k],
            fixed=mask | (contact_of >= 0),
            rejected_edges=self.rejected_edges[k],
            before=self.before[k],
            contact_of=contact_of,
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
            before=self.before[k],
            bend=block.bend(points) if moved else block.tight_start,
        )

    def via_result(self, k: int, *, best: bool) -> ViaResult:
        """Via k's best centre, with `best`, else its input."""
        x, y = self.best_positions[k] if best else self.start_positions[k]
        return ViaResult((float(x), float(y)), self.via_rejected[k])

    def step(self) -> float:
        """One burst of every member and via together; returns its largest
        change of a shape or a via's centre.

        Logs at DEBUG what the burst cost: L-BFGS-B's iterations and energy
        evaluations, the seconds those evaluations took and the seconds of
        the rest of the step, and what L-BFGS-B says ended it. The seconds
        are wall time by `time.perf_counter`, not by the run's clock. The
        rest includes L-BFGS-B's own work and the after-burst check, which
        the benchmark's "outside L-BFGS-B" splits differently."""
        began = time.perf_counter()
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
        contact_of: list[IntArray] = []
        pressed = self.pressed(self.points, self.positions, self.seen)
        for i, (points, mask, kept) in enumerate(
            zip(self.points, self.masks, self.contact_of, strict=True)
        ):
            if i in self.free and not self.resampled[i]:
                points, mask, kept = _resample(
                    points, mask, kept, self.loose[i], settings.spacing, pressed[i]
                )
            starts.append(points)
            masks.append(mask)
            contact_of.append(kept)
        self.masks, self.contact_of = masks, contact_of
        self.resampled = [False] * len(starts)
        burst = self.burst(starts)
        change = max(
            [
                *(shape_change(before[i], burst.polylines[i]) for i in self.free),
                *(
                    float(np.linalg.norm(after - before))
                    for after, before in zip(
                        burst.positions, self.positions, strict=True
                    )
                ),
            ]
        )
        self.moved |= change > NOTHING_MOVED
        self.points, self.positions = burst.polylines, burst.positions
        self.seen = self.seeing(self.points)
        for i, (count, close) in enumerate(
            zip(burst.rejected_edges, burst.too_close, strict=True)
        ):
            self.rejected_edges[i] += count
            self.too_close[i] += close
        for k, count in enumerate(burst.via_rejected):
            self.via_rejected[k] += count
        self.energy = self.energy_of(self.points, self.positions, self.seen)
        if self.energy <= self.best_energy and self.guard(
            self.points, self.masks, self.positions, self.seen
        ):
            self.best, self.best_masks = list(self.points), list(self.masks)
            self.best_contact_of = list(self.contact_of)
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
        self.settled = self.gave_up or (
            not burst.abnormal
            and (
                (not cut_short and change < settings.settled_change)
                or stalled(self.lowest, settings)
            )
        )
        rest = time.perf_counter() - began - burst.evaluating
        logger.debug(
            f"Burst of {self.description}: {burst.iterations} L-BFGS-B iterations, "
            f"{burst.evaluations} evaluations, {burst.evaluating:.3f} s in "
            f"evaluations, {rest:.3f} s in the rest of the step, "
            f'ended with "{burst.message}"'
        )
        return change

    def pressed(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> list[BoolArray]:
        """Per member, its vertices at `polylines` more than PRESSED_DEPTH
        inside the clearance of the other members and vias at
        `positions`, which resampling keeps where they are, as free
        vertices that the next burst moves. Spread along a shape that the
        free vertices beside them bend, they would creep out of the
        clearance a little every burst, and the run would settle later.
        Vertices deep inside static copper are left to resampling, which
        lays them there where a chord cuts a corner and moves them out
        again. `seen` is what the members see at `polylines`."""
        flat = shown_flat(self.members, polylines, positions)
        return [
            distance < -PRESSED_DEPTH
            for distance in seen.distances(polylines, flat, -PRESSED_DEPTH)
        ]

    def layout(self, starts: Sequence[FloatArray]) -> _Layout:
        """The variables of a burst from `starts` and the current centres."""
        index, count = _variables(self.masks, self.groups)
        offsets = [np.zeros_like(points) for points in starts]
        via_index = count + np.arange(len(self.vias), dtype=np.intp)
        for k, via in zip(via_index, self.vias, strict=True):
            for i, end, offset in via.carried:
                index[i][end] = k
                offsets[i][end] = offset
        count += len(self.vias)
        return _Layout(
            list(starts), list(self.positions), index, offsets, via_index, count
        )

    def objective(
        self,
        layout: _Layout,
        seen: Seen | None = None,
        listed: bool = True,
    ) -> Callable[[FloatArray], tuple[float, FloatArray]]:
        """The problem's energy as a function of `layout`'s variables, with
        what each member sees of the others measured at the layout's start
        unless given, through a `NeighbourList` unless not `listed`."""
        if seen is None:
            seen = self.seeing(layout.starts)
        masks = self.masks
        neighbours = NeighbourList(seen, self.reach, LIST_SKIN) if listed else None

        def objective(x: FloatArray) -> tuple[float, FloatArray]:
            polylines, positions = layout.unflatten(x)
            value, gradients, via_gradients = self.total(
                polylines, positions, masks, seen, neighbours
            )
            return value, layout.gradient(gradients, via_gradients)

        return objective

    def burst(self, starts: list[FloatArray]) -> _ProblemBurst:
        """One L-BFGS-B run over every free vertex and moving via of the
        problem, with what moved back where edges newly crossed static
        copper or another member's new shape, a section newly bent below its
        limit or brought its parts beyond a turn too close
        (`polylines.SelfCheck`) or a via ended newly inside its clearance."""
        members = self.members
        positions = self.positions
        layout = self.layout(starts)
        index, count = layout.index, layout.count
        free = [i >= 0 for i in index]
        seen = self.seeing(starts)
        objective = self.objective(layout, seen)
        x0 = layout.x0
        flat_start = shown_flat(members, starts, positions)
        static_before = [
            body.field.segments_clear(s[:-1], s[1:])
            for body, s in zip(self.bodies, starts, strict=True)
        ]
        others_before = seen.segments_clear(starts, flat_start)
        # Only a vertex that newly leaves its contact moves back, so that
        # moving back always ends.
        left_before = [self.left(i, s) for i, s in enumerate(starts)]
        checks = [
            SelfCheck.of(s, member.limit, member.apart)
            if isinstance(member, PairBlock)
            else None
            for member, s in zip(members, starts, strict=True)
        ]
        clear_before = [
            distance >= -INSIDE_TOLERANCE
            for distance in self.via_distances(starts, positions)
        ]
        stopped_before = self.stop_distances(starts, positions)

        def crossings(x: FloatArray) -> tuple[_Round, BoolArray]:
            """What `_Round` counts at `x`, and the coordinates of the
            variables to move back."""
            polylines, centres = layout.unflatten(x)
            flat = shown_flat(members, polylines, centres)
            back = np.zeros(count, dtype=bool)
            counts: list[tuple[int, int]] = []
            via_counts = [0] * len(self.vias)
            others = seen.segments_clear(polylines, flat)
            for i, points in enumerate(polylines):
                a, b = points[:-1], points[1:]
                crossing = (
                    static_before[i] & ~self.bodies[i].field.segments_clear(a, b)
                ) | (others_before[i] & ~others[i])
                ends = np.zeros(len(points), dtype=bool)
                ends[:-1] |= crossing
                ends[1:] |= crossing
                for k, end in self.carried[i]:
                    via_counts[k] += int(ends[end])
                outside = self.left(i, points) & ~left_before[i]
                ends |= outside
                rejected, close = int(crossing.sum()) + int(outside.sum()), 0
                check = checks[i]
                if check is not None:
                    own, more, close = check(points)
                    ends |= own
                    rejected += more
                counts.append((rejected, close))
                back[index[i][ends & free[i]]] = True
            for k, (distance, stopped) in enumerate(
                zip(
                    self.via_distances(polylines, centres),
                    self.stop_distances(polylines, centres),
                    strict=True,
                )
            ):
                if (clear_before[k] and distance < -INSIDE_TOLERANCE) or (
                    stopped < min(stopped_before[k], -INSIDE_TOLERANCE)
                ):
                    back[layout.via_index[k]] = True
                    via_counts[k] += 1
            return _Round(counts, via_counts), np.repeat(back, 2)

        found: list[_Round] = []

        def check(candidate: FloatArray) -> tuple[int, BoolArray]:
            found_now, back = crossings(candidate)
            if not found:
                found.append(found_now)
            return sum(r for r, _ in found_now.members) + sum(found_now.vias), back

        def revert(x: FloatArray) -> int:
            return revert_until_clear(x, x0, check)[1]

        evaluating = 0.0

        def timed(x: FloatArray) -> tuple[float, FloatArray]:
            nonlocal evaluating
            began = time.perf_counter()
            answer = objective(x)
            evaluating += time.perf_counter() - began
            return answer

        result = run_burst(timed, x0, revert, self.settings)
        polylines, centres = layout.unflatten(result.points)
        # An abnormally ended burst stops at its check without counting it.
        if result.abnormal or not found:
            counted = _Round([(0, 0)] * len(starts), [0] * len(self.vias))
        else:
            counted = found[0]
        return _ProblemBurst(
            polylines,
            centres,
            [rejected for rejected, _ in counted.members],
            [close for _, close in counted.members],
            list(counted.vias),
            result.abnormal,
            result.iterations,
            result.evaluations,
            result.message,
            evaluating,
        )


def _description(layers: Sequence[str], vias: bool) -> str:
    """How log lines and results name a problem of `layers`, with moving
    `vias` or without."""
    named = (
        layers[0] if len(layers) == 1 else f"{', '.join(layers[:-1])} and {layers[-1]}"
    )
    return f"the chains and vias on {named}" if vias else f"the chains on {named}"


def _no_deeper(distances: Sequence[float], starts: Sequence[float]) -> bool:
    """Whether each of `distances` is no deeper inside than its start, and
    inside."""
    return all(
        distance >= min(start, -INSIDE_TOLERANCE)
        for distance, start in zip(distances, starts, strict=True)
    )


def _one_net(via: _Via, other: _Via) -> bool:
    return via.given.via.net == other.given.via.net
