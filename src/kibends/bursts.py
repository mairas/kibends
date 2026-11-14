"""One L-BFGS burst, which the optimizer runs over every free coordinate
of a problem at once, and the optimizer's settings, which every module
importing them takes from here.

The burst runs in Rust (`kibends_core::burst`): argmin's L-BFGS ended by
scipy's L-BFGS-B rules, evaluating a `kibends._core.Objective` without
calling Python. Between iterations it checks for signals and asks the
caller's `stop`, so that Ctrl-C and the time limit end it there.

A burst does not bound its coordinates; a check after the run moves back
whatever newly crosses an obstacle. The run loop (`kibends_core::optimize`)
runs each of its bursts through `optimize.run_burst`, which is this
module's, with its check a `kibends._core.Revert`.

A burst whose objective is a `ScaledObjective` minimises over y, with
x = s ⊙ y for the objective's per-coordinate scale s, so that L-BFGS's one
curvature model fits stiff and soft coordinates alike. The scale travels
with the objective rather than as an argument of its own, so that whatever
wraps `run_burst` and passes the objective on keeps it; the check and the
burst's result stay in x.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

from kibends import _core
from kibends.energy import FloatArray

FALLBACK_WARNING = (
    "A burst's crossings named nothing more to move back; moving the whole burst back"
)
"""Logged when a burst's check moves the whole burst back
(`kibends_core::optimize::checks::revert_until_clear`)."""

INSIDE_TOLERANCE = _core.INSIDE_TOLERANCE
"""Depth inside the clearance, mm, that the violation guard ignores as float
noise: a track routed at exactly the clearance splits into vertices about
1e-15 mm inside. Far below KiCad's 1 nm coordinate step."""

MAX_COARSE_LEVELS = _core.MAX_COARSE_LEVELS
"""The most coarse spacings a schedule may hold before its final spacing."""

# Edges shorter than this fraction of the final spacing meet the short-edge
# barrier, at every level. Drawn edges shorter than that keep their vertices
# until the first burst slides them apart.
SHORT_EDGE_FRACTION = 0.1

# The chain's length term is ill-conditioned (tension spreads one vertex per
# iteration), so scipy's default gtol (1e-5) ends a burst long before the
# shape settles; with this gtol, burst_maxiter or ftol ends it instead. ftol is
# near scipy's default (2.2e-9): it ends a few bursts early and lets problems
# settle in fewer bursts, while 1e-12 and below save nothing measurable, as
# measured in https://github.com/mairas/kibends/issues/70.
LBFGSB_FTOL = 1e-9
LBFGSB_GTOL = 1e-10

DEFAULT_CHORD_TOLERANCE = 0.01
"""How far the committed tracks may depart from the optimised curve, mm
(`simplify`): well below fabrication tolerances; at 5 µm a 23 mm chain on a
real board came out as 63 tracks, and at 0.025 mm the committed tracks
strayed visibly from the optimised curve."""


@dataclass(frozen=True)
class OptimizerSettings:
    spacing: float = 0.1
    """The longest edge between a chain's vertices at the final level, mm."""
    coarse_spacings: tuple[float, ...] = (1.6, 0.4)
    """The spacings a problem runs at before `spacing`, mm, coarsest first:
    each above the next and the last above `spacing`, at most
    `MAX_COARSE_LEVELS` of them. Empty runs at `spacing` alone.

    A long track bowed across its whole span barely moves per burst at
    0.1 mm among many other tracks; with few vertices it straightens in a
    few bursts. On a region of a real board, 92 chains and 70 moving vias in
    one problem, 1.6 and 0.4 took 31 iterations and 8.7 s where 0.1 alone
    took 113 and 45 s and left a 45 mm track bowed, and the whole board
    ran best with them among the schedules compared
    (https://github.com/mairas/kibends/issues/184)."""
    settled_change: float = DEFAULT_CHORD_TOLERANCE / 2
    """A problem whose burst moves no shape and no via's centre by this much,
    mm, settles. Half the chord tolerance, the run's own unless
    `--settled-change` is given (`cli.smooth_options`): a smaller move
    cannot change the committed tracks by more than simplification already
    may."""
    max_iterations: int = 200
    """Outer iterations after which the run stops unconverged and keeps each
    problem's best state. This and the stall rule were measured in
    https://github.com/mairas/kibends/issues/27 and
    https://github.com/mairas/kibends/issues/28."""
    stall_tolerance: float = 1e-3
    """A problem settles when, over `stall_window` of its bursts, its best
    energy improved by less than this for each moving member and via and
    its tracks shortened by less than this in all (`optimize.stalled`).
    The energy is in mm of track length, so this is 1 µm of length: in
    flat stretches the shapes keep wandering by tens of µm while the
    energy creeps down by far less."""
    stall_window: int = 3
    burst_maxiter: int = 50
    """L-BFGS iterations per burst of a problem. Shorter bursts lay out
    and check the shapes more often, which on a large problem reaches a
    given length in fewer energy evaluations.

    Smoothing is a visual task that does not need the exact optimum, and
    almost every burst runs to this cap, so the cap sets the cost: at 50
    with `lbfgs_ftol` 1e-9 the scenarios take 44% fewer energy evaluations
    than at 100 with 1e-15 and end 30 µm longer in sum. At 25 a coupled
    section settles visibly short of its shape: the stall rule counts
    bursts, so its window shrinks with this. Measured in
    https://github.com/mairas/kibends/issues/70."""
    lbfgs_ftol: float = LBFGSB_FTOL
    """The relative energy reduction of an L-BFGS iteration at or below
    which a burst ends."""
    lbfgs_gtol: float = LBFGSB_GTOL
    """The gradient's largest component at or below which a burst ends."""
    time_limit: float | None = None
    """Seconds each level of the schedule may run, after which the run ends
    the running burst at its next L-BFGS iteration. Every problem then at a
    coarse level goes on at the next, the limit counted again; with none
    there, the run keeps its best state, as at the iteration cap
    (`optimize.optimize_chains`). None for no limit."""

    def __post_init__(self) -> None:
        if self.spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        levels = (*self.coarse_spacings, self.spacing)
        if any(finer >= coarser for coarser, finer in pairwise(levels)):
            raise ValueError(
                "coarse_spacings must fall strictly, coarsest first, to above "
                f"spacing {self.spacing}, got {self.coarse_spacings}"
            )
        if len(self.coarse_spacings) > MAX_COARSE_LEVELS:
            raise ValueError(
                f"coarse_spacings holds at most {MAX_COARSE_LEVELS} spacings, "
                f"got {len(self.coarse_spacings)}"
            )

    @property
    def short_edge(self) -> float:
        """Length below which an edge meets the energy's short-edge barrier, mm."""
        return SHORT_EDGE_FRACTION * self.spacing


DEFAULT_SETTINGS = OptimizerSettings()


Check = Callable[[FloatArray], int]
"""What a burst runs after L-BFGS, such as a `kibends._core.Revert`: it moves
back, in place, what newly crosses an obstacle and returns how many edges
crossed."""
Stop = Callable[[], bool]


class ScaledObjective:
    """An objective, answering as itself, and the scale of each coordinate
    that a burst of it minimises over (`run_burst`)."""

    def __init__(self, objective: _core.Objective, scale: FloatArray) -> None:
        self.objective = objective
        self.scale = scale

    def __call__(self, x: FloatArray) -> tuple[float, FloatArray]:
        return self.objective(x)


Objective = _core.Objective | ScaledObjective


@dataclass(frozen=True)
class Burst:
    points: FloatArray
    """Where the burst left its flat coordinates."""
    abnormal: bool
    iterations: int = 0
    """The L-BFGS iterations it ran."""
    evaluations: int = 0
    """The times L-BFGS evaluated the objective."""
    message: str = ""
    """What ended it, as scipy's L-BFGS-B words it; empty for a burst that
    did not run."""
    cut: bool = False
    """Whether `stop` ended it before its iteration cap."""
    evaluating: float = 0.0
    """Seconds the evaluations took."""


def run_burst(
    objective: Objective,
    start: FloatArray,
    revert: Check,
    settings: OptimizerSettings,
    stop: Stop | None = None,
) -> Burst:
    """Minimise `objective` from `start` with L-BFGS, then move back what
    `revert` finds.

    `revert` moves back, in place, the coordinates of edges that newly cross
    an obstacle, and returns how many edges crossed. `stop`, asked after
    each L-BFGS iteration, ends the run there when it answers True, such
    as when the time limit has passed. With no coordinate to move, the
    burst ends where it is without running L-BFGS. A `ScaledObjective`
    runs L-BFGS over the coordinates divided by its scale. The burst
    carries what L-BFGS reports of its run; one whose iteration failed
    ends abnormally at the last point an iteration reached
    (`kibends_core::burst`).
    """
    if len(start) == 0:
        return Burst(start.copy(), False)
    if isinstance(objective, ScaledObjective):
        core, scale = objective.objective, objective.scale
    else:
        core, scale = objective, None
    result = _core.run_burst(
        core,
        start,
        scale,
        settings.burst_maxiter,
        settings.lbfgs_ftol,
        settings.lbfgs_gtol,
        stop,
    )
    current = result.points
    revert(current)
    return Burst(
        current,
        result.abnormal,
        result.iterations,
        result.evaluations,
        result.message,
        result.cut,
        result.evaluating,
    )
