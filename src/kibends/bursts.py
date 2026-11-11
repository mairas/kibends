"""One L-BFGS-B burst, which the optimizer runs over every free coordinate
of a problem at once, and the optimizer's settings, which every module
importing them takes from here.

A burst does not bound its coordinates; a check after the run moves back
whatever newly crosses an obstacle.

A burst whose objective is a `ScaledObjective` minimises over y, with
x = s ⊙ y for the objective's per-coordinate scale s, so that L-BFGS-B's
one curvature model fits stiff and soft coordinates alike. The scale
travels with the objective rather than as an argument of its own, so that
whatever wraps `run_burst` and passes the objective on keeps it; the check
and the burst's result stay in x.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from kibends.energy import BoolArray, DistanceField, FloatArray

FALLBACK_WARNING = (
    "A burst's crossings named nothing more to move back; moving the whole burst back"
)

# Depth inside the clearance, mm, that the violation guard ignores as float
# noise: a track routed at exactly the clearance resamples to vertices about
# 1e-15 mm inside. Far below KiCad's 1 nm coordinate step.
INSIDE_TOLERANCE = 1e-7

# How much deeper than at the start something already inside may measure
# without going deeper, mm: round-off in measuring a shape that moved along
# what it is inside, such as an edge sliding inside a clearance, up to
# about 1e-16 mm. A thousandth of KiCad's 1 nm coordinate step.
DEEPER_ROUND_OFF = 1e-9

LBFGSB_ABNORMAL = 2

# Edges shorter than this fraction of the spacing meet the short-edge barrier.
# Resampling leaves every edge at a free vertex at least half the spacing.
SHORT_EDGE_FRACTION = 0.1

# The chain's length term is ill-conditioned (tension spreads one vertex per
# iteration), so L-BFGS-B's default gtol (1e-5) ends a burst long before the
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
    """Target distance between resampled vertices, mm."""
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
    """L-BFGS-B iterations per burst of a problem. Shorter bursts resample
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
    """L-BFGS-B's relative energy reduction below which a burst ends."""
    lbfgs_gtol: float = LBFGSB_GTOL
    """L-BFGS-B's projected gradient below which a burst ends."""
    time_limit: float | None = None
    """Seconds after which the run ends the running burst at its next
    L-BFGS-B iteration and keeps its best state, as at the iteration cap;
    None for no limit."""

    def __post_init__(self) -> None:
        if self.spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    @property
    def short_edge(self) -> float:
        """Length below which an edge meets the energy's short-edge barrier, mm."""
        return SHORT_EDGE_FRACTION * self.spacing


DEFAULT_SETTINGS = OptimizerSettings()


def point_distance(field: DistanceField, point: FloatArray) -> float:
    return float(field.signed_distance(point[None])[0][0])


Objective = Callable[[FloatArray], tuple[float, FloatArray]]
Revert = Callable[[FloatArray], int]
Stop = Callable[[], bool]
Crossings = Callable[[FloatArray], tuple[int, BoolArray]]


class ScaledObjective:
    """An objective, answering as itself, and the scale of each coordinate
    that a burst of it minimises over (`run_burst`)."""

    def __init__(self, objective: Objective, scale: FloatArray) -> None:
        self.objective = objective
        self.scale = scale

    def __call__(self, x: FloatArray) -> tuple[float, FloatArray]:
        return self.objective(x)

    def over_scaled(self, y: FloatArray) -> tuple[float, FloatArray]:
        """The objective and its gradient by y at x = scale ⊙ y."""
        value, gradient = self.objective(self.scale * y)
        return value, self.scale * gradient


@dataclass(frozen=True)
class Burst:
    points: FloatArray
    """Where the burst left its flat coordinates."""
    abnormal: bool
    iterations: int = 0
    """The L-BFGS-B iterations it ran."""
    evaluations: int = 0
    """The times L-BFGS-B evaluated the objective."""
    message: str = ""
    """What L-BFGS-B says ended it; empty for a burst that did not run."""
    cut: bool = False
    """Whether `stop` ended it before its iteration cap."""


def revert_until_clear(
    candidate: FloatArray, start: FloatArray, crossings: Crossings
) -> tuple[BoolArray, int]:
    """Move coordinates of `candidate` back to `start`, in place, until
    `crossings` finds no edge newly crossing an obstacle.

    `crossings` gives the number of crossings and the coordinates to move
    back for them. Moving back can make a neighbouring edge cross, hence
    the loop; it ends because the start, judged against itself, crosses
    nothing. A crossing whose moving coordinates are all back already
    moves the whole candidate back, with `FALLBACK_WARNING` logged. Returns
    the mask of coordinates moved back and the number of crossings in the
    candidate as given.
    """
    reverted = np.zeros(len(candidate), dtype=bool)
    initial: int | None = None
    while True:
        count, back = crossings(candidate)
        if initial is None:
            initial = count
        if count == 0:
            return reverted, initial
        back &= ~reverted
        if not back.any():
            # A crossing whose moving coordinates are all back already, so
            # that what crosses does not move back. Moving everything back
            # ends the loop, as the start crosses nothing.
            logger.warning(FALLBACK_WARNING)
            back = ~reverted
        candidate[back] = start[back]
        reverted |= back


def run_burst(
    objective: Objective,
    start: FloatArray,
    revert: Revert,
    settings: OptimizerSettings,
    stop: Stop | None = None,
) -> Burst:
    """Minimise `objective` from `start` with L-BFGS-B, then move back what
    `revert` finds.

    `revert` moves back, in place, the coordinates of edges that newly cross
    an obstacle, and returns how many edges crossed. `stop`, asked after
    each L-BFGS-B iteration, ends the run there when it answers True, such
    as when the time limit has passed. With no coordinate to move, the
    burst ends where it is without running L-BFGS-B. A `ScaledObjective`
    runs L-BFGS-B over the coordinates divided by its scale. The burst
    carries what L-BFGS-B reports of its run.
    """
    if len(start) == 0:
        # L-BFGS-B would return a result without a status.
        return Burst(start.copy(), False)
    halted = False

    # scipy's callback types and dispatch key on this parameter name.
    def asked(intermediate_result: object) -> None:
        nonlocal halted
        if stop is not None and stop():
            halted = True
            raise StopIteration

    if isinstance(objective, ScaledObjective):
        scale = objective.scale
        posed, y0 = objective.over_scaled, start / scale
    else:
        scale = None
        posed, y0 = objective, start.copy()
    result = minimize(
        posed,
        y0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": settings.burst_maxiter,
            "ftol": settings.lbfgs_ftol,
            "gtol": settings.lbfgs_gtol,
        },
        callback=asked,
    )
    current = np.array(result.x, dtype=float)
    if scale is not None:
        current *= scale
    revert(current)
    cut = halted and result.nit < settings.burst_maxiter
    return Burst(
        current,
        result.status == LBFGSB_ABNORMAL,
        result.nit,
        result.nfev,
        result.message,
        cut,
    )
