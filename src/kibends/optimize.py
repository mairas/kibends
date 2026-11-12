"""The optimizer: every chain, coupled section and moving via of a problem
as one L-BFGS problem.

The run loop is Rust's (`kibends._core.advance`, `kibends._core.ProblemState`;
the design is in `kibends_core::optimize` and its modules `state`, `checks`
and `layout`). This module sets a run up: it splits the members into
problems (`_problems`), reads each problem's members and vias once into its
Rust state (`_Problem`), and between the Rust calls asks `to_hold`, builds
again the problems whose sections it holds and logs what the run reports.
Each burst runs through `bursts.run_burst`, its evaluations in Rust.

What does not depend on a problem lives beside it: `members` (the members
as the energy takes them), `polylines` (statistics), `contacts` (the terms
that keep a vertex on its contact) and `results` (what a run returns).
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from loguru import logger

from kibends import _core
from kibends.bursts import (
    DEFAULT_SETTINGS,
    FALLBACK_WARNING,
    INSIDE_TOLERANCE,
    Burst,
    OptimizerSettings,
    ScaledObjective,
    run_burst,
)
from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    IntArray,
    ObstacleField,
    obstacle_reach,
)
from kibends.members import Member, body_of, core_members, positions_array
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainCentreline,
    core_field,
)
from kibends.pair_blocks import Hold, PairBlock, PairInput, pair_block
from kibends.polylines import measure
from kibends.results import (
    ChainResult,
    OptimizeResult,
    PairResult,
    RunResult,
    Termination,
    ViaResult,
)
from kibends.seen import Seen, ViaSight
from kibends.via_moves import FreeDistances, MovingVias
from kibends.via_pairs import ViaPairs
from kibends.vias import StaticVia, ViaInput

MAX_ABNORMAL_BURSTS = _core.MAX_ABNORMAL_BURSTS
"""After this many abnormal L-BFGS ends in a row a problem counts as
settled and keeps what its bursts reached."""

PRESSED_DEPTH = _core.PRESSED_DEPTH
"""How deep inside the other members' and vias' clearance, mm, a vertex
lies when resampling keeps it where it is."""

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
        else measure(chain.polyline, settings.spacing)
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


def stalled(
    lowest_energies: Sequence[float],
    lengths: Sequence[float],
    settings: OptimizerSettings,
    members: int,
    vias: int = 0,
) -> bool:
    """Whether a problem of `members` moving members and `vias` moving vias
    has stalled: over `stall_window` bursts its lowest energy fell by no
    more than `stall_tolerance` for each member and via, and its tracks'
    summed `lengths` by no more than one tolerance
    (`kibends_core::optimize::stalled`)."""
    return _core.stalled(
        list(lowest_energies),
        list(lengths),
        settings.stall_window,
        settings.stall_tolerance,
        members,
        vias,
    )


End = tuple[int, int]
"""A chain end: the chain's index and 0 or -1."""


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
    static_vias: Sequence[StaticVia] = (),
    to_hold: Callable[[Sequence[PairResult]], Sequence[int]] | None = None,
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
    `settled_change` and moved nothing back, or when the lowest energy its
    bursts reached, whether or not the guard took them as its best state,
    and its tracks' length stalled over `stall_window` bursts (`stalled`),
    or after MAX_ABNORMAL_BURSTS abnormal bursts in a row; the run converges
    when every problem has, and gives up (`Termination.GAVE_UP`) if some
    problem settled on abnormal bursts. A run that gave up on some problem
    and whose best states sum to no less energy than its inputs ends
    `Termination.ABNORMAL`, whatever stopped it. With a
    time limit in `settings`, the run stops once `clock` reads the limit or
    more past `started`, by default when the run starts: the running burst
    ends at its next L-BFGS iteration, and its state goes through the
    checks and the guard as any burst's. A run stopped, by either, while
    some problem runs at a coarse level ends each such problem at the final
    spacing (`ProblemState.finish`) and says so
    (`RunResult.before_final_spacing`).

    Whenever the run would end, `to_hold` is given each section's best
    state and names the sections to hold (`Hold.CROWDING`). Each problem
    with such a section takes it back to its drawn shape and goes on from
    its best state (`_Problem.go_on_from`), with the members and vias that
    would come too close to the section, or to what was put back, put back
    where they started; the others keep their progress. The iteration cap
    counts again from the hold. A run that would end at the time limit
    runs no further burst: it asks `to_hold` again after each hold, since
    what a hold put back may be bent too tightly, until it names no
    section that is not held.

    It logs at INFO when setup begins, once the problems are built, the
    setup time and the starting length and energy, as each problem enters a
    level of a schedule with coarse levels its spacing (`_stage`), and at
    each hold what it put back. Each outer iteration
    logs its progress at DEBUG, and at INFO once `PROGRESS_INTERVAL` has
    passed since `started` or the last INFO line. Each burst logs what it
    cost at DEBUG only (`_burst_done`).
    """
    logger.info("Setting up the optimizer")
    set_up = clock()
    junctions(chains)
    blocks = [pair_block(pair, settings.spacing) for pair in pairs]
    everything = [*chains, *(block.chain for block in blocks)]
    moving_vias = MovingVias(everything, vias, hole_clearance, hole_to_hole)
    placed = _problems(chains, blocks, moving_vias, params, settings, static_vias)
    problems = [p.problem for p in placed]
    moving = [p for p in problems if p.moving]
    # A problem given up on before a hold stays given up on, though the
    # problem that replaces it converges.
    gave_up: dict[str, int] = {}

    def sections(*, best: bool) -> tuple[PairResult, ...]:
        by_section = {
            i: p.problem.pair_result(len(p.chains) + k, best=best)
            for p in placed
            for k, i in enumerate(p.sections)
        }
        return tuple(by_section[k] for k in range(len(blocks)))

    def result(termination: Termination, iterations: int, *, best: bool) -> RunResult:
        by_index = {
            i: p.problem.result(k, best=best)
            for p in placed
            for k, i in enumerate(p.chains)
        }
        by_via = {
            v: p.problem.via_result(k, best=best)
            for p in placed
            for k, v in enumerate(p.vias)
        }
        given_up = dict(gave_up)
        for p in problems:
            if p.gave_up:
                given_up.setdefault(p.description, p.abnormal_streak)
        return RunResult(
            tuple(by_index[i] for i in range(len(chains))),
            termination,
            iterations,
            vias=tuple(by_via[v] for v in range(len(vias))),
            pairs=sections(best=best),
            gave_up=tuple(given_up.items()),
            abnormal=tuple(
                (p.description, p.abnormal_bursts)
                for p in problems
                if p.abnormal_bursts
            ),
        )

    if not moving:
        return result(Termination.NOTHING_TO_DO, 0, best=False)
    begun = clock()
    logger.info(_starting(moving, begun - set_up))
    origin = begun if started is None else started
    deadline = None if settings.time_limit is None else origin + settings.time_limit

    def ended() -> bool:
        return deadline is not None and clock() >= deadline

    expired = None if deadline is None else ended
    report = _Report(moving, clock, origin, begun)

    def burst(
        objective: _core.Objective,
        start: FloatArray,
        scale: FloatArray,
        revert: _core.Revert,
    ) -> Burst:
        return run_burst(
            ScaledObjective(objective, scale), start, revert, settings, expired
        )

    iteration = 0
    cap = settings.max_iterations
    termination = Termination.ITERATION_CAP
    while True:
        # Past the time limit, what a hold puts back is judged again, with
        # no burst, until it leaves no section to hold.
        if termination is not Termination.TIME_LIMIT:
            outcome, iteration = _core.advance(
                [p.state for p in moving],
                iteration,
                cap,
                bool(gave_up),
                expired,
                burst,
                report,
            )
            termination = Termination(outcome)
            if termination is Termination.NOTHING_TO_DO:
                return result(Termination.NOTHING_TO_DO, iteration, best=False)
        named: Sequence[int] = () if to_hold is None else to_hold(sections(best=True))
        held = [k for k in named if not blocks[k].held]
        if not held:
            break
        for k in held:
            blocks[k] = pair_block(
                replace(blocks[k].given, hold=Hold.CROWDING), settings.spacing
            )
        for n, p in enumerate(placed):
            here = [k for k in held if k in p.sections]
            if not here:
                continue
            if p.problem.gave_up:
                gave_up.setdefault(p.problem.description, p.problem.abnormal_streak)
            problem = _problem(
                chains,
                blocks,
                moving_vias,
                params,
                settings,
                static_vias,
                layers=p.layers,
                on=p.chains,
                sections=p.sections,
                joined=p.vias,
            )
            members, vias_back = problem.go_on_from(p.problem)
            logger.info(
                f"Holding {len(here)} coupled section{'' if len(here) == 1 else 's'} "
                f"of {problem.description} as drawn and going on; put back as "
                f"they started: {members} other member{'' if members == 1 else 's'} "
                f"and {vias_back} via{'' if vias_back == 1 else 's'}"
            )
            placed[n] = replace(p, problem=problem)
        problems = [p.problem for p in placed]
        moving = [p for p in problems if p.moving]
        report.problems = moving
        cap = iteration + settings.max_iterations
    # A run stopped at a coarse level still ends at the final spacing.
    before_final = any(p.state.level < len(settings.coarse_spacings) for p in moving)
    for p in moving:
        p.state.finish()
    final = replace(
        result(termination, iteration, best=True),
        before_final_spacing=before_final,
    )
    if final.gave_up and sum(p.best_energy for p in moving) >= sum(
        p.input_energy for p in moving
    ):
        return replace(final, termination=Termination.ABNORMAL)
    return final


_Notice = (
    _core.FallbackNotice
    | _core.AbnormalNotice
    | _core.BurstNotice
    | _core.StageNotice
    | _core.ProgressNotice
)


class _Report:
    """What the run tells as it goes, logged: a fallback as
    `FALLBACK_WARNING`, an abnormal burst and each burst's cost at DEBUG
    (`_burst_done`), and the progress after each outer iteration at DEBUG,
    and at INFO once `PROGRESS_INTERVAL` has passed by `clock` since
    `shown`, the last INFO line. A problem entering a level is logged at
    INFO (`_stage`). The
    notices name `problems` by their place; the elapsed seconds count from
    `origin`."""

    def __init__(
        self,
        problems: Sequence[_Problem],
        clock: Callable[[], float],
        origin: float,
        shown: float,
    ) -> None:
        self.problems = problems
        self.clock = clock
        self.origin = origin
        self.shown = shown

    def __call__(self, notice: _Notice) -> None:
        if isinstance(notice, _core.FallbackNotice):
            logger.warning(FALLBACK_WARNING)
        elif isinstance(notice, _core.AbnormalNotice):
            logger.debug(
                "L-BFGS ended a burst of "
                f"{self.problems[notice.problem].description} abnormally, "
                f"{notice.streak} in a row"
            )
        elif isinstance(notice, _core.BurstNotice):
            _burst_done(self.problems[notice.problem], notice)
        elif isinstance(notice, _core.StageNotice):
            logger.info(_stage(self.problems[notice.problem], notice.level))
        else:
            now = self.clock()
            message = _progress(notice, now - self.origin)
            if now - self.shown >= PROGRESS_INTERVAL:
                logger.info(message)
                self.shown = now
            else:
                logger.debug(message)


def _level_spacing(settings: OptimizerSettings, level: int) -> float:
    """The spacing of `level` of the schedule: its coarse spacings, then the
    final one."""
    coarse = settings.coarse_spacings
    return coarse[level] if level < len(coarse) else settings.spacing


def _stage(problem: _Problem, level: int) -> str:
    """The line that says `problem` enters `level`, a coarse one or the
    final one, and its spacing."""
    spacing = _level_spacing(problem.settings, level)
    if level < len(problem.settings.coarse_spacings):
        return f"Optimizing {problem.description} coarsely, at {spacing:g} mm spacing"
    return f"Refining {problem.description} at {spacing:g} mm spacing"


def _burst_done(problem: _Problem, notice: _core.BurstNotice) -> None:
    """Logs at DEBUG what a burst of `problem` cost: L-BFGS's iterations and
    energy evaluations, the seconds those evaluations took and the seconds
    of the rest of the step, and what L-BFGS says ended it, and the spacing
    of a coarse level it ran at. The seconds are wall time, not by the run's
    clock. The rest includes L-BFGS's own work and the after-burst check,
    which the benchmark's "outside L-BFGS" splits differently."""
    settings = problem.settings
    coarse = (
        f", at {_level_spacing(settings, notice.level):g} mm spacing"
        if notice.level < len(settings.coarse_spacings)
        else ""
    )
    logger.debug(
        f"Burst of {problem.description}: {notice.iterations} L-BFGS iterations, "
        f"{notice.evaluations} evaluations, {notice.evaluating:.3f} s in "
        f"evaluations, {notice.rest:.3f} s in the rest of the step, "
        f'ended with "{notice.message}"{coarse}'
    )


@dataclass(frozen=True)
class _Placed:
    """A problem and the indices of its chains, sections and vias among
    those of the run."""

    problem: _Problem
    layers: list[str]
    chains: list[int]
    sections: list[int]
    vias: list[int]


def _problem(
    chains: Sequence[ChainInput],
    blocks: Sequence[PairBlock],
    vias: MovingVias,
    params: EnergyParams,
    settings: OptimizerSettings,
    static_vias: Sequence[StaticVia],
    *,
    layers: Sequence[str],
    on: Sequence[int],
    sections: Sequence[int],
    joined: Sequence[int],
) -> _Problem:
    """The problem of `layers`: the chains `on` them, the `sections` and
    the `joined` vias, by their indices in `chains`, `blocks` and `vias`."""
    members = [*on, *(len(chains) + k for k in sections)]
    local = {i: n for n, i in enumerate(members)}
    return _Problem(
        layers,
        [chains[i] for i in on],
        [blocks[k] for k in sections],
        [_problem_via(vias, v, local, joined) for v in joined],
        params,
        settings,
        static_vias,
        hole_clearance=vias.hole_clearance,
        hole_to_hole=vias.hole_to_hole,
    )


def _problems(
    chains: Sequence[ChainInput],
    blocks: Sequence[PairBlock],
    vias: MovingVias,
    params: EnergyParams,
    settings: OptimizerSettings,
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
        joined = [
            v
            for v, via in enumerate(vias.vias)
            if group is tied or any(via.via.reaches(layer) for layer in group)
        ]
        on = [i for i, c in enumerate(chains) if c.layer in group]
        sections = [k for k, b in enumerate(blocks) if b.given.layer in group]
        problem = _problem(
            chains,
            blocks,
            vias,
            params,
            settings,
            static_vias,
            layers=group,
            on=on,
            sections=sections,
            joined=joined,
        )
        placed.append(_Placed(problem, group, on, sections, joined))
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


def _starting(problems: Sequence[_Problem], setup: float) -> str:
    """One line on the run about to start, after `setup` seconds of setup."""
    count = len(problems)
    return (
        f"Optimizer set up in {setup:.1f} s: "
        f"{count} problem{'' if count == 1 else 's'}, "
        f"length {sum(p.length() for p in problems):.3f} mm, "
        f"energy {sum(p.energy for p in problems):.3f}"
    )


def _progress(notice: _core.ProgressNotice, elapsed: float) -> str:
    """One line on where the run stands after an outer iteration."""
    return (
        f"Iteration {notice.iteration}, {elapsed:.1f} s: "
        f"length {sum(notice.lengths):.3f} mm, "
        f"energy {sum(notice.energies):.3f}, "
        f"largest change {notice.change:.2e} mm"
    )


def _one_net(via: _Via, other: _Via) -> bool:
    return via.given.via.net == other.given.via.net


def _description(layers: Sequence[str], vias: bool) -> str:
    """How log lines and results name a problem of `layers`, with moving
    `vias` or without."""
    named = (
        layers[0] if len(layers) == 1 else f"{', '.join(layers[:-1])} and {layers[-1]}"
    )
    return f"the chains and vias on {named}" if vias else f"the chains on {named}"


class _Problem:
    """The chains, coupled sections and vias of the layers that moving vias
    tie together, or of one layer, read once into their Rust state
    (`kibends._core.ProblemState`), with what the Python side builds them
    from. The chains come first, then the sections, as their centrelines."""

    def __init__(
        self,
        layers: Sequence[str],
        chains: Sequence[ChainInput],
        sections: Sequence[PairBlock],
        vias: Sequence[_Via],
        params: EnergyParams,
        settings: OptimizerSettings,
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
        self.description = _description(layers, bool(vias))
        groups = junctions(chains)
        count = len(chains) + len(sections)
        self.carried: list[list[tuple[int, int]]] = [[] for _ in range(count)]
        for k, via in enumerate(vias):
            for i, end, _ in via.carried:
                self.carried[i].append((k, end))
        self.rings = [via.rings for via in vias]
        apart = [via.apart for via in vias]
        # Vias of one net only stop each other; the pairs of the pairwise
        # term are summed apart from the smooth minimum; the rest repel.
        own_net = [
            {u: d for u, d in a.items() if _one_net(vias[k], vias[u])}
            for k, a in enumerate(apart)
        ]
        rings = [dict(r) for r in self.rings]
        self.via_sight = ViaSight(
            rings,
            [
                {u: d for u, d in a.items() if u not in own}
                for a, own in zip(apart, own_net, strict=True)
            ],
            count,
        )
        paired = pairs.paired
        self.via_repel = ViaSight(
            rings,
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
        self.core = _core.Problem(
            core_members(self.members),
            rings,
            [via.given.via.diameter for via in vias],
            [core_field(via.given.field) for via in vias],
            [
                None if (stops := via.given.stops) is None else core_field(stops)
                for via in vias
            ],
            self.via_repel,
            self.via_stop,
            None if self.pairs is None else self.pairs.core,
            params,
            settings.short_edge,
        )
        """The problem's energy and what its members see of each other
        (`kibends_core::optimize::problem`)."""
        self.before = [
            measure(body.centreline.points, settings.spacing) for body in self.bodies
        ]
        self.state = _core.ProblemState(
            self.core,
            [body.centreline.points for body in self.bodies],
            [body.fixed for body in self.bodies],
            [[v for v, _ in body.contacts] for body in self.bodies],
            [body.held for body in self.bodies],
            [m.limit if isinstance(m, PairBlock) else None for m in self.members],
            positions_array([np.array(v.given.via.position) for v in vias]),
            [
                [(i, end == -1, (float(o[0]), float(o[1]))) for i, end, o in v.carried]
                for v in vias
            ],
            [core_field(v.given.copper) for v in vias],
            [
                [None if f is None else core_field(f) for f in v.given.stop_parts]
                for v in vias
            ],
            [[(i, end == -1) for i, end in group] for group in groups],
            self.via_sight,
            settings.spacing,
            settings.coarse_spacings,
            settings.settled_change,
            settings.stall_window,
            settings.stall_tolerance,
            settings.burst_maxiter,
            settings.lbfgs_ftol,
            settings.lbfgs_gtol,
        )
        """Its shapes, best state and progress (`kibends_core::optimize::state`)."""

    @property
    def moving(self) -> bool:
        return self.state.moving

    @property
    def free(self) -> list[int]:
        """The members that move."""
        return self.state.free

    @property
    def moved(self) -> bool:
        return self.state.moved

    @property
    def gave_up(self) -> bool:
        """Whether it settled on MAX_ABNORMAL_BURSTS abnormal bursts in a row."""
        return self.state.gave_up

    @property
    def abnormal_streak(self) -> int:
        return self.state.abnormal_streak

    @property
    def abnormal_bursts(self) -> int:
        return self.state.abnormal_bursts

    @property
    def energy(self) -> float:
        return self.state.energy

    @property
    def best_energy(self) -> float:
        return self.state.best_energy

    @property
    def input_energy(self) -> float:
        return self.state.input_energy

    @property
    def points(self) -> list[FloatArray]:
        return self.state.points

    @property
    def masks(self) -> list[BoolArray]:
        return self.state.masks

    @property
    def contact_of(self) -> list[IntArray]:
        return self.state.contact_of

    @property
    def positions(self) -> list[FloatArray]:
        return list(self.state.positions)

    @property
    def seen(self) -> Seen:
        """What the members see of each other at `points`."""
        return self.state.seen

    @property
    def allowed(self) -> list[int]:
        """Per member, the violations of its resampled start, which the
        guard holds it to."""
        return self.state.allowed

    def length(self) -> float:
        """The summed length of the selected tracks at their current shapes,
        a section's as its two tracks, mm."""
        return self.state.length()

    def go_on_from(self, earlier: _Problem) -> tuple[int, int]:
        """Take `earlier`'s best state, `earlier` being this problem with
        fewer sections held, but keep the sections held only here as drawn,
        and put back where they started the members and vias that would
        then come too close (`kibends_core::optimize::state::State::go_on_from`).
        Returns how many members that move and how many vias went back."""
        return self.state.go_on_from(earlier.state)

    def seeing(self, polylines: Sequence[FloatArray]) -> Seen:
        """What each member sees of the others and of the vias with the
        members at `polylines` (`members.seen_parts`)."""
        return self.core.seen(list(polylines))

    def guard(
        self,
        polylines: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> bool:
        """Whether nothing oversteps at `polylines` and `positions`: no
        member has more violations than `allowed`, and no via is newly or
        deeper inside its clearance or its stops than it started. `seen` is
        what the members see at `polylines`."""
        return self.state.guard(
            list(polylines), list(masks), positions_array(positions), seen
        )

    def violations(
        self,
        polylines: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> list[int]:
        """Per member at `polylines`, its `polylines.count_violations` of its
        static copper, the other members and the vias at `positions`
        together, every member's at once through `seen`."""
        return self.state.violations(
            list(polylines), list(masks), positions_array(positions), seen
        )

    def pressed(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> list[BoolArray]:
        """Per member, its vertices at `polylines` more than PRESSED_DEPTH
        inside the clearance of the other members and vias at `positions`,
        which resampling keeps where they are."""
        return self.state.pressed(list(polylines), positions_array(positions), seen)

    def via_distances(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        below: float = -INSIDE_TOLERANCE,
    ) -> FreeDistances:
        """Each moving via's free distance to each obstacle of the copper of
        other nets: its static copper, and those below `below` of the
        members near it at `polylines` and the vias of other nets at
        `positions`."""
        return self.state.via_distances(
            list(polylines), positions_array(positions), below
        )

    def via_energy(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
    ) -> tuple[float, list[FloatArray], FloatArray]:
        """The moving vias' own obstacle terms and their gradient at every
        vertex and at every via's centre
        (`kibends_core::optimize::problem::Problem::via_energy`)."""
        return self.core.via_energy(list(polylines), positions_array(positions))

    def via_stiffness(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
    ) -> FloatArray:
        """Each moving via's stiffness under its own obstacle terms."""
        return self.core.via_stiffness(list(polylines), positions_array(positions))

    def energy_masks(self, masks: Sequence[BoolArray]) -> list[BoolArray]:
        """`masks` with the ends the moving vias carry loose."""
        loose = [mask.copy() for mask in masks]
        for mask, carried in zip(loose, self.carried, strict=True):
            for _, end in carried:
                mask[end] = False
        return loose

    def layout(self, starts: Sequence[FloatArray]) -> _core.BurstLayout:
        """The variables of a burst from `starts` and the current centres."""
        return self.state.layout(list(starts))

    def entered(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        seen: Seen,
    ) -> list[tuple[int, int]]:
        """Each (member, member or via) whose clearance the member's edges
        at `polylines` enter, the vias at `positions`, numbered after the
        members. `seen` is what the members see at `polylines`."""
        return self.state.entered(list(polylines), positions_array(positions), seen)

    def failing(
        self,
        polylines: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        positions: Sequence[FloatArray],
        seen: Seen,
        at_start: Sequence[tuple[int, int]],
    ) -> BoolArray:
        """For each member, then each via, whether at `polylines` and
        `positions` it oversteps, enters a member or via that it did not
        enter at the start, `at_start` (`entered`), or is entered so, as
        `go_on_from` judges what to put back."""
        return self.state.failing(
            list(polylines),
            list(masks),
            positions_array(positions),
            seen,
            list(at_start),
        )

    def lookups(self, seen: Seen) -> _core.Lookups:
        """The neighbour list and the static lists made for `seen`, for the
        evaluations of one objective."""
        return self.core.lookups(seen)

    def objective(
        self, layout: _core.BurstLayout, seen: Seen, lookups: _core.Lookups | None
    ) -> _core.Objective:
        """The problem's energy as a function of `layout`'s variables, with
        what each member sees of the others as `seen`, searching through
        `lookups`, by default each evaluation its own search."""
        return self.core.objective(
            layout.starts,
            layout.index,
            layout.offsets,
            layout.via_index,
            layout.count,
            self.energy_masks(self.masks),
            self.contact_of,
            seen,
            lookups,
        )

    def scale(
        self, starts: Sequence[FloatArray], seen: Seen, lookups: _core.Lookups | None
    ) -> FloatArray:
        """Each flat coordinate's scale for a burst from `starts`
        (`kibends_core::optimize::layout::scale`). `seen` and `lookups` are
        as for `objective`."""
        return self.state.scale(list(starts), seen, lookups)

    def result(self, k: int, *, best: bool) -> ChainResult:
        """Chain k's best state, with `best`, else its input."""
        polyline, mask, contact_of, rejected = self.state.member(k, best)
        return ChainResult(
            polyline=polyline,
            fixed=mask | (contact_of >= 0),
            rejected_edges=rejected,
            before=self.before[k],
            contact_of=contact_of,
        )

    def pair_result(self, k: int, *, best: bool) -> PairResult:
        """Member k's, a section's, best state, with `best`, else its input."""
        block = self.members[k]
        assert isinstance(block, PairBlock)
        given = block.given
        moved = best and not block.held
        points, mask, _, rejected = self.state.member(k, best)
        a, b = block.offset(points) if best else (given.a.points, given.b.points)
        return PairResult(
            block=block,
            centreline=points,
            fixed=mask & ~block.mergeable_pins(points) if moved else mask,
            a=a,
            b=b,
            rejected_edges=rejected,
            before=self.before[k],
            bend=block.bend(points) if moved else block.tight_start,
        )

    def via_result(self, k: int, *, best: bool) -> ViaResult:
        """Via k's best centre, with `best`, else its input."""
        position, rejected = self.state.via(k, best)
        return ViaResult(position, rejected)
