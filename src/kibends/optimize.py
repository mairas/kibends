"""The optimizer: every chain, coupled section and moving via of a run as
one L-BFGS problem.

The run loop is Rust's (`kibends._core.advance`, `kibends._core.ProblemState`;
the design is in `kibends_core::optimize` and its modules `state`, `checks`
and `layout`). This module sets a run up: it gathers the members into the
run's problem (`_problem_of`), reads its members and vias once into its Rust
state (`_Problem`), and between the Rust calls asks `to_hold`, builds the
problem again when it holds sections and logs what the run reports.
Each burst runs through `bursts.run_burst`, its evaluations in Rust.

What does not depend on a problem lives beside it: `members` (the members
as the energy takes them), `polylines` (statistics), `contacts` (the terms
that keep a vertex on its contact) and `results` (what a run returns).

Nothing in the energy keeps a chain's ends apart, so a moving via may shrink
a chain it carries to nothing, as a fan-out whose via reaches its pad's
anchor. The result then moves the via by at most MIN_TRACK_LENGTH, so that
the chain commits as a track (`_Problem.via_shift`).
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

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
from kibends.chain import NM_PER_MM
from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    IntArray,
    ObstacleField,
    obstacle_reach,
)
from kibends.inside import (
    NO_CHORDS,
    START_TOLERANCE,
    ArcChords,
    InsideClearanceError,
    Rules,
    Violation,
    violations,
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


MIN_TRACK_LENGTH = math.sqrt(2) / NM_PER_MM
"""The shortest chain a moving via leaves, mm: its ends, each rounded to
KiCad's 1 nm grid, stay apart. KiCad takes a track whose ends coincide for
null and deletes it in its track cleanup, and its footprint drag, carrying
the track ends on a pad's anchor, would leave a via with no track to it
behind."""


# A run logs its progress at INFO at most this often, seconds, and not at all
# if it ends sooner; DEBUG gets every outer iteration.
PROGRESS_INTERVAL = 5.0


@dataclass(frozen=True)
class Progress:
    """Where a run stands after an outer iteration, or at iteration 0 where
    it starts."""

    elapsed: float
    """Seconds by the run's clock since it started."""
    iteration: int
    energy: float
    """Every problem's energy, summed."""
    length: float
    """Every problem's track length, summed, mm: the selected tracks at
    their current shapes, before simplification."""
    level: int
    """The level of the schedule the run stands at, counted from 0: the
    coarsest level some problem still runs at, as problems may run at
    different levels."""
    levels: int
    """The schedule's levels: its coarse spacings, then the final one."""


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


class StoppedInSetupError(Exception):
    """`stop` answered true while the optimizer set its problems up:
    nothing optimized."""


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
    stop: Callable[[], bool] | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    arc_chords: ArcChords = NO_CHORDS,
) -> RunResult:
    """Minimise the chains', sections' and vias' summed energy jointly, as
    one problem on every layer they are on.

    Takes chains whose ends are fixed or loose where they meet at junctions
    (`junctions`), coupled sections of pairs (`pairs`) and moving `vias`,
    and returns each chain's, section's and via's best state; a junction's
    ends come back at one point, loose. A via keeps `hole_clearance` and
    `hole_to_hole` where holes are involved. `static_vias` are the board's
    vias that are none of `vias`; each problem keeps those its moving vias
    share a layer with.

    `stop` is asked in setup after building each problem and after
    checking where the copper starts, and raises `StoppedInSetupError` if
    it answers true there, since setup runs no burst whose state could be
    kept. A problem's build is not stopped within.

    Before any burst it raises `inside.InsideClearanceError` naming each
    place where the copper of a problem that moves starts inside a
    clearance, allowing copper on a selected arc's concave side the arc's
    offset from its chords (`arc_chords`, `inside`): a burst keeps the
    clearance its start keeps, and the run is to end clean.

    Each problem runs through `settings`' schedule: its chains at each of
    `coarse_spacings` in turn, then at the final spacing, a level handing
    over when the problem settles at it (`kibends_core::optimize::state`).
    Each problem keeps its best state, replaced only after a
    burst that does not raise its energy, leaves no chain or section with
    more violations than it started with, and leaves no via deeper inside
    its clearance or its stops than it started. It goes on at the final
    spacing from the coarse levels' best state only if some coarse burst
    ended normally and that state costs no more than its inputs; else from
    the inputs.

    A problem settles when its burst moved no shape or via by
    `settled_change` and moved nothing back, or when the lowest energy its
    bursts reached, whether or not the guard took them as its best state,
    and its tracks' length stalled over `stall_window` bursts (`stalled`),
    or after MAX_ABNORMAL_BURSTS abnormal bursts in a row, at a coarse level
    handing over instead; the run converges
    when every problem has, and gives up (`Termination.GAVE_UP`) if some
    problem settled on abnormal bursts. A run that gave up on some problem
    and whose best states sum to no less energy than its inputs ends
    `Termination.ABNORMAL`, whatever stopped it. With a
    time limit in `settings`, the limit runs out once `clock` reads it or
    more past the end of setup, when optimizing starts, and no problem owes
    a burst at its level (`_core.OwingNotice`): one that has not
    settled and has run none at a level a handover took it to. While some
    problem owes one, a burst past the limit runs until it accepts its
    first L-BFGS iteration and ends there, cut, so that each level a
    handover took a problem to descends; the problems after it in the
    outer iteration step so too. Once the limit has run out, the
    running burst ends before its next line-search trial, at its last
    accepted point (`bursts.run_burst`), and its state goes
    through the checks and the guard as any burst's. Each problem then
    still before the final level goes on at its next level as if it had
    settled where it runs, one that has taken no step leaving level 0, and
    the limit counts again from then (`_hand_over`); the rest go on as they
    were. With none before the final level the run ends
    `Termination.TIME_LIMIT`. So every level a problem reaches above level
    0 runs a burst, and the limit counts at most once for each level,
    besides the overrun of the bursts running when it runs out, about an
    L-BFGS iteration for each owed one, and the holds. Should it run out
    in the outer iteration that
    reaches the iteration cap with some problem before the final level,
    the cap ends the run; after a hold the limit is still out, so the run
    goes on as when it runs out, with no burst first.
    `stop`, asked whenever the time limit is, ends the running burst the
    same way and the run at once, `Termination.STOPPED`, if it answered
    true before the limit ran out. A run that `stop` or the iteration cap
    ends while some problem runs at a coarse level ends each such problem
    at the final spacing (`ProblemState.finish`), and one that has taken no
    step at its inputs; either way it says so
    (`RunResult.before_final_spacing`).

    Whenever the run would end, `to_hold` is given each section's best
    state and names the sections to hold (`Hold.CROWDING`). Each problem
    with such a section takes it back to its drawn shape and goes on from
    its best state (`_Problem.go_on_from`), with the members and vias that
    would come too close to the section, or to what was put back, put back
    where they started; the others keep their progress. The iteration cap
    counts again from the hold. A run that would end at the time limit or
    on `stop`, even in the iteration it converged, runs no further burst
    and ends on whichever ended it: it asks `to_hold` again after each hold, since
    what a hold put back may be bent too tightly, until it names no
    section that is not held.

    It logs at INFO when setup begins, once the problems are built, the
    setup time and the starting length and energy, as each problem enters a
    level of a schedule with coarse levels its spacing (`_stage`), each
    time the time limit ends a coarse level its spacing and the problems
    that left it (`_hand_over`),
    and at each hold what it put back. Each
    outer iteration
    logs its progress at DEBUG, and at INFO once `PROGRESS_INTERVAL` has
    passed since optimizing started or the last INFO line, and goes to
    `on_progress` as a `Progress`, its seconds counted from `started`, by
    default when optimizing starts; `on_progress` first hears the start, as
    iteration 0, once the problems are set up. Each burst logs what it cost
    at DEBUG only (`_burst_done`).
    """
    logger.info("Setting up the optimizer")
    set_up = clock()
    junctions(chains)
    blocks = [pair_block(pair, settings.spacing) for pair in pairs]
    everything = [*chains, *(block.chain for block in blocks)]
    moving_vias = MovingVias(everything, vias, hole_clearance, hole_to_hole)

    def stopped_in_setup() -> None:
        if stop is not None and stop():
            raise StoppedInSetupError

    if not _layers(moving_vias):
        stopped_in_setup()
        return RunResult((), Termination.NOTHING_TO_DO, 0)
    problem = _problem_of(chains, blocks, moving_vias, params, settings, static_vias)
    stopped_in_setup()
    if problem.moving:
        unclean = problem.start_violations(arc_chords)
        if unclean:
            raise InsideClearanceError(unclean)
    stopped_in_setup()
    # Given up on before a hold, the problem stays given up on, though the
    # problem that replaces it converges.
    gave_up: tuple[str, int] | None = None

    def sections(*, best: bool) -> tuple[PairResult, ...]:
        return tuple(
            problem.pair_result(len(chains) + k, best=best) for k in range(len(blocks))
        )

    def result(termination: Termination, iterations: int, *, best: bool) -> RunResult:
        given_up = gave_up
        if given_up is None and problem.gave_up:
            given_up = (problem.description, problem.abnormal_streak)
        return RunResult(
            tuple(problem.result(i, best=best) for i in range(len(chains))),
            termination,
            iterations,
            vias=tuple(problem.via_result(v, best=best) for v in range(len(vias))),
            pairs=sections(best=best),
            gave_up=() if given_up is None else (given_up,),
            abnormal=(
                ((problem.description, problem.abnormal_bursts),)
                if problem.abnormal_bursts
                else ()
            ),
        )

    if not problem.moving:
        return result(Termination.NOTHING_TO_DO, 0, best=False)
    begun = clock()
    logger.info(_starting(problem, begun - set_up))
    origin = begun if started is None else started
    limit = settings.time_limit
    deadline = None if limit is None else begun + limit

    # Which of `stop` and the time limit answered true first, once one has.
    # Cast, as the checker would otherwise take it as None where `ended`
    # has set it.
    first_end = cast(Termination | None, None)
    # Whether the problem owes a burst at its level, as the run last said
    # (`_core.OwingNotice`): a burst then runs past the limit to its first
    # accepted iteration.
    owing = False

    def owe(now: bool) -> None:
        nonlocal owing
        owing = now

    def ended(iterated: bool = False) -> bool:
        """Whether `stop` or the time limit ends the run, the limit while
        the problem owes a burst only once the running one has `iterated`."""
        nonlocal first_end
        if stop is not None and stop():
            first_end = first_end or Termination.STOPPED
            return True
        if deadline is not None and clock() >= deadline and (iterated or not owing):
            first_end = first_end or Termination.TIME_LIMIT
            return True
        return False

    expired = None if deadline is None and stop is None else ended
    report = _Report(problem, clock, origin, begun, owe, on_progress)
    report.start()

    def burst(
        objective: _core.Objective,
        start: FloatArray,
        scale: FloatArray,
        revert: _core.Revert,
    ) -> Burst:
        return run_burst(
            ScaledObjective(objective, scale, revert), start, revert, settings, expired
        )

    def advance(iteration: int, cap: int) -> tuple[Termination, int]:
        """`_core.advance` from `iteration` until `cap`, the time limit
        counted again whenever it ends a coarse level (`_hand_over`),
        unless at `cap`, which then ends the run."""
        nonlocal deadline, first_end
        while True:
            if not problem.moving:
                # A hold left nothing free.
                return Termination.NOTHING_TO_DO, iteration
            report.read_level()
            outcome, iteration = _core.advance(
                problem.state,
                iteration,
                cap,
                gave_up is not None,
                expired,
                burst,
                report,
            )
            termination = Termination(outcome)
            if (
                termination is not Termination.TIME_LIMIT
                or first_end is not Termination.TIME_LIMIT
                or limit is None
                or not problem.state.before_final
            ):
                return termination, iteration
            if iteration >= cap:
                # The cap ends the run, so that a hold after it goes on as
                # after any cap.
                first_end = None
                return Termination.ITERATION_CAP, iteration
            _hand_over(problem)
            deadline = clock() + limit
            first_end = None

    iteration = 0
    cap = settings.max_iterations
    termination = Termination.ITERATION_CAP
    while True:
        # Past the time limit, what a hold puts back is judged again, with
        # no burst, until it leaves no section to hold.
        if termination is not Termination.TIME_LIMIT:
            termination, iteration = advance(iteration, cap)
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
        if gave_up is None and problem.gave_up:
            gave_up = (problem.description, problem.abnormal_streak)
        rebuilt = _problem_of(
            chains, blocks, moving_vias, params, settings, static_vias
        )
        members, vias_back = rebuilt.go_on_from(problem)
        logger.info(
            f"Holding {len(held)} coupled section{'' if len(held) == 1 else 's'} "
            f"of {rebuilt.description} as drawn and going on; put back as "
            f"they started: {members} other member{'' if members == 1 else 's'} "
            f"and {vias_back} via{'' if vias_back == 1 else 's'}"
        )
        problem = rebuilt
        report.problem = problem
        cap = iteration + settings.max_iterations
        # A stop or the time limit in the iteration that converged still
        # ends the run, as the hold has undone the convergence.
        if first_end is not None:
            termination = Termination.TIME_LIMIT
    # Within the loop a stop counts as the time limit, so that a hold after
    # it starts no burst.
    if termination is Termination.TIME_LIMIT and first_end is Termination.STOPPED:
        termination = Termination.STOPPED
    # A run stopped at a coarse level still ends at the final spacing.
    before_final = problem.moving and problem.state.before_final
    if problem.moving:
        problem.state.finish()
    final = replace(
        result(termination, iteration, best=True),
        before_final_spacing=before_final,
    )
    if final.gave_up and (
        not problem.moving or problem.best_energy >= problem.input_energy
    ):
        return replace(final, termination=Termination.ABNORMAL)
    return final


_Notice = (
    _core.FallbackNotice
    | _core.AbnormalNotice
    | _core.BurstNotice
    | _core.StageNotice
    | _core.OwingNotice
    | _core.ProgressNotice
)


class _Report:
    """What the run tells as it goes, logged: a fallback as
    `FALLBACK_WARNING`, an abnormal burst and each burst's cost at DEBUG
    (`_burst_done`), and the progress after each outer iteration at DEBUG,
    and at INFO once `PROGRESS_INTERVAL` has passed by `clock` since
    `shown`, the last INFO line, and handed to `on_progress`, which first
    hears the start (`start`). The problem entering a level is logged at
    INFO (`_stage`), and whether it owes a burst at its level goes to
    `on_owing`, and kept with the level read before the run advances
    (`read_level`) to hand on in the progress. The elapsed seconds count
    from `origin`."""

    def __init__(
        self,
        problem: _Problem,
        clock: Callable[[], float],
        origin: float,
        shown: float,
        on_owing: Callable[[bool], None],
        on_progress: Callable[[Progress], None] | None = None,
    ) -> None:
        self.problem = problem
        self.clock = clock
        self.origin = origin
        self.shown = shown
        self.on_owing = on_owing
        self.on_progress = on_progress
        self.level = 0
        self.read_level()

    def read_level(self) -> None:
        """Take the problem's level from its state, which `_core.advance`
        holds while it reports: before each call to it, the level it
        reports from then on as `_core.StageNotice`s."""
        self.level = self.problem.state.level

    def start(self) -> None:
        """Hand the run's start, at `shown`, to `on_progress` as iteration 0,
        at level 0: the problem stands at its inputs, laid out at the final
        spacing, and its first step enters the coarsest level."""
        self._hand_on(
            self.shown - self.origin,
            0,
            self.problem.energy,
            self.problem.length(),
            0,
        )

    def _hand_on(
        self,
        elapsed: float,
        iteration: int,
        energy: float,
        length: float,
        level: int,
    ) -> None:
        if self.on_progress is None:
            return
        self.on_progress(
            Progress(
                elapsed,
                iteration,
                energy,
                length,
                level,
                len(self.problem.settings.coarse_spacings) + 1,
            )
        )

    def __call__(self, notice: _Notice) -> None:
        if isinstance(notice, _core.FallbackNotice):
            logger.warning(FALLBACK_WARNING)
        elif isinstance(notice, _core.AbnormalNotice):
            logger.debug(
                "L-BFGS ended a burst of "
                f"{self.problem.description} abnormally, "
                f"{notice.streak} in a row"
            )
        elif isinstance(notice, _core.BurstNotice):
            _burst_done(self.problem, notice)
        elif isinstance(notice, _core.OwingNotice):
            self.on_owing(notice.owing)
        elif isinstance(notice, _core.StageNotice):
            self.level = notice.level
            logger.info(_stage(self.problem, notice.level))
        else:
            now = self.clock()
            elapsed = now - self.origin
            message = _progress(notice, elapsed)
            if now - self.shown >= PROGRESS_INTERVAL:
                logger.info(message)
                self.shown = now
            else:
                logger.debug(message)
            self._hand_on(
                elapsed,
                notice.iteration,
                notice.energy,
                notice.length,
                self.level,
            )


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


def _hand_over(problem: _Problem) -> None:
    """The problem, before the final level, goes on at the next, as if it
    had settled where it runs (`ProblemState.hand_over`). Logs at INFO the
    level's spacing and the next one's."""
    level = problem.state.hand_over()
    if level is None:
        return
    settings = problem.settings
    logger.info(
        f"Time limit reached at {_level_spacing(settings, level):g} mm spacing; "
        f"going on at {_level_spacing(settings, level + 1):g} mm spacing "
        f"with {problem.description}"
    )


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


def _layers(vias: MovingVias) -> list[str]:
    """The run's layers: the chains' layers, those a via reaches first, or,
    when no via reaches one, the vias' own layers first (stitching vias
    selected alone); none for a run with nothing to move. `vias` holds the
    chains, then the sections' centrelines."""
    chain_layers = list(dict.fromkeys(chain.layer for chain in vias.chains))
    reached = [
        layer
        for layer in chain_layers
        if any(via.via.reaches(layer) for via in vias.vias)
    ]
    if not reached:
        reached = list(
            dict.fromkeys(layer for via in vias.vias for layer in via.via.span)
        )
    return list(dict.fromkeys([*reached, *chain_layers]))


def _problem_of(
    chains: Sequence[ChainInput],
    blocks: Sequence[PairBlock],
    vias: MovingVias,
    params: EnergyParams,
    settings: OptimizerSettings,
    static_vias: Sequence[StaticVia] = (),
) -> _Problem:
    """The run's problem on its `_layers`: every chain, section and moving
    via, with the `static_vias` that share a layer with a moving via."""
    return _Problem(
        _layers(vias),
        chains,
        blocks,
        [_problem_via(vias, v) for v in range(len(vias.vias))],
        params,
        settings,
        static_vias,
        hole_clearance=vias.hole_clearance,
        hole_to_hole=vias.hole_to_hole,
    )


def _problem_via(vias: MovingVias, v: int) -> _Via:
    """Via v of `vias` in the run's problem."""
    apart: dict[int, float] = {}
    for u in range(len(vias.vias)):
        if u == v:
            continue
        inflation = vias.apart(v, u)
        if inflation is not None:
            apart[u] = inflation
    return _Via(
        vias.vias[v],
        tuple((e.chain, e.index, e.offset) for e in vias.ends[v]),
        dict(vias.rings(v)),
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


def _starting(problem: _Problem, setup: float) -> str:
    """One line on the run about to start, after `setup` seconds of setup."""
    return (
        f"Optimizer set up in {setup:.1f} s: "
        f"length {problem.length():.3f} mm, energy {problem.energy:.3f}"
    )


def _progress(notice: _core.ProgressNotice, elapsed: float) -> str:
    """One line on where the run stands after an outer iteration."""
    return (
        f"Iteration {notice.iteration}, {elapsed:.1f} s: "
        f"length {notice.length:.3f} mm, "
        f"energy {notice.energy:.3f}, "
        f"largest change {notice.change:.2e} mm"
    )


def _one_net(via: _Via, other: _Via) -> bool:
    return via.given.via.net == other.given.via.net


def _listed(names: Sequence[str]) -> str:
    """`names` as a list in a sentence: "a", "a and b", "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _description(layers: Sequence[str], vias: bool) -> str:
    """How log lines and results name a problem of `layers`, with moving
    `vias` or without."""
    named = _listed(layers)
    return f"the chains and vias on {named}" if vias else f"the chains on {named}"


class _Problem:
    """The run's chains, coupled sections and vias, read once into their Rust
    state (`kibends._core.ProblemState`), with what the Python side builds
    them from. The chains come first, then the sections, as their
    centrelines."""

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
        self.hole_clearance = hole_clearance
        self.hole_to_hole = hole_to_hole
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
        """Per member, the violations of its inputs, which the guard holds
        it to."""
        return self.state.allowed

    def length(self) -> float:
        """The summed length of the selected tracks at their current shapes,
        a section's as its two tracks, mm."""
        return self.state.length()

    def start_violations(self, chords: ArcChords) -> list[Violation]:
        """Where its inputs, the drawn copper, are inside a clearance by the
        rules a user can see, named (`inside`), each place on the concave
        side of the arc `chords` allowed the arc's offset from them."""
        return violations(
            self.state.start_inside(START_TOLERANCE),
            self.members,
            [v.given for v in self.vias],
            Rules(self.hole_clearance, self.hole_to_hole, chords),
        )

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
        """Chain k's best state, with `best`, its ends moved with their
        vias' `via_shift`, else its input."""
        polyline, mask, contact_of, rejected = self.state.member(k, best)
        if best:
            polyline = polyline.copy()
            for v, end in self.carried[k]:
                polyline[end] += self.via_shift(v)
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
        """Via k's best centre, with `best`, moved by its `via_shift`, else
        its input."""
        (x, y), rejected = self.state.via(k, best)
        if best:
            dx, dy = self.via_shift(k)
            x, y = x + float(dx), y + float(dy)
        return ViaResult((x, y), rejected)

    def via_shift(self, k: int) -> FloatArray:
        """How far via k moves from its best centre so that no chain it
        carries alone, with its other end staying, is shorter than
        MIN_TRACK_LENGTH: its end of that chain goes to MIN_TRACK_LENGTH
        from the other end, toward where it was drawn, so that the
        direction comes from the drawn geometry and not from what rounding
        leaves of a chain shrunk to nothing. A chain drawn with both ends on
        one point aims along its end's offset now instead, and one shrunk
        to exactly that point as well keeps it. Zero for most."""
        for i, end, _ in self.vias[k].carried:
            if len(self.carried[i]) > 1:
                continue
            points = self.state.member(i, True)[0]
            other = points[-1 if end == 0 else 0]
            if float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) >= (
                MIN_TRACK_LENGTH
            ):
                continue
            toward: FloatArray = self.state.member(i, False)[0][end] - other
            if not toward.any():
                toward = points[end] - other
            if not toward.any():
                continue
            unit = toward / float(np.linalg.norm(toward))
            return other + MIN_TRACK_LENGTH * unit - points[end]
        return np.zeros(2)
