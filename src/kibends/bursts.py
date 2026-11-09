"""One L-BFGS-B burst, which the optimizer runs over every free coordinate
of a problem at once, and the optimizer's settings, which every module
importing them takes from here.

A burst does not bound its coordinates; a check after the run moves back
whatever newly crosses an obstacle.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from kibends.energy import BoolArray, DistanceField, FloatArray

# Depth inside the clearance, mm, that the violation guard ignores as float
# noise: a track routed at exactly the clearance resamples to vertices about
# 1e-15 mm inside. Far below KiCad's 1 nm coordinate step.
INSIDE_TOLERANCE = 1e-7

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


@dataclass(frozen=True)
class OptimizerSettings:
    spacing: float = 0.1
    """Target distance between resampled vertices, mm."""
    frozen_depth: float = 0.01
    """δ, mm: vertices and vias that start deeper than δ inside an
    obstacle's clearance are frozen; resampling also keeps vertices a few δ
    inside the other members' clearance where they are."""
    settled_change: float = 1e-3
    """A problem whose burst moves no shape and no via's centre by this much,
    mm, settles."""
    max_iterations: int = 200
    stall_tolerance: float = 1e-3
    """A problem settles when its best energy improved by less than this
    over `stall_window` of its bursts. The energy is in mm of track length,
    so this is 1 µm of length: in flat stretches the shapes keep wandering
    by tens of µm while the energy creeps down by far less."""
    stall_window: int = 3
    burst_maxiter: int = 50
    """L-BFGS-B iterations per burst of a problem. Shorter bursts resample
    and check the shapes more often, which on a large problem reaches a
    given length in fewer energy evaluations. At 25 a coupled
    section settles visibly short of its shape, as measured in
    https://github.com/mairas/kibends/issues/70: the stall rule counts
    bursts, so its window shrinks with this."""
    lbfgs_ftol: float = LBFGSB_FTOL
    """L-BFGS-B's relative energy reduction below which a burst ends."""
    lbfgs_gtol: float = LBFGSB_GTOL
    """L-BFGS-B's projected gradient below which a burst ends."""
    time_limit: float | None = None
    """Seconds after which the run stops at the next burst's end and keeps
    its best state, as at the iteration cap; None for no limit."""

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
Crossings = Callable[[FloatArray], tuple[int, BoolArray]]


@dataclass(frozen=True)
class Burst:
    points: FloatArray
    """Where the burst left its flat coordinates."""
    abnormal: bool


def revert_until_clear(
    candidate: FloatArray, start: FloatArray, crossings: Crossings
) -> tuple[BoolArray, int]:
    """Move coordinates of `candidate` back to `start`, in place, until
    `crossings` finds no edge newly crossing an obstacle.

    `crossings` gives the number of crossing edges and the coordinates to
    move back for them. Moving back can make a neighbouring edge cross,
    hence the loop; it ends because edges moved back whole are the start's,
    clear ones. Returns the mask of coordinates moved back and the number of
    edges that crossed in the candidate as given.
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
            # Should not happen: an edge with both ends moved back is as
            # clear as at the start. Should it, moving everything back ends
            # the loop, as the start crosses nothing.
            back = ~reverted
        candidate[back] = start[back]
        reverted |= back


def run_burst(
    objective: Objective,
    start: FloatArray,
    revert: Revert,
    settings: OptimizerSettings,
) -> Burst:
    """Minimise `objective` from `start` with L-BFGS-B, then move back what
    `revert` finds.

    `revert` moves back, in place, the coordinates of edges that newly cross
    an obstacle, and returns how many edges crossed. With no coordinate to
    move, the burst ends where it is.
    """
    if len(start) == 0:
        # L-BFGS-B would return a result without a status.
        return Burst(start.copy(), False)
    result = minimize(
        objective,
        start.copy(),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": settings.burst_maxiter,
            "ftol": settings.lbfgs_ftol,
            "gtol": settings.lbfgs_gtol,
        },
    )
    current = np.array(result.x, dtype=float)
    revert(current)
    return Burst(current, result.status == LBFGSB_ABNORMAL)
