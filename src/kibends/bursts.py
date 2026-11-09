"""One bounded L-BFGS-B burst, which the optimizer runs over every free
coordinate of a problem at once, and the optimizer's settings, which every
module importing them takes from here.

Every coordinate moves only inside its bubble's box, so nothing jumps over
an obstacle; a check after the run moves back whatever newly crosses one,
and the run is repeated with halved bounds for what moved back.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from kibends.energy import BoolArray, DistanceField, FloatArray

# Depth inside the clearance, mm, that the violation guard ignores as float
# noise: a track routed at exactly the clearance resamples to vertices about
# 1e-15 mm inside. Far below KiCad's 1 nm coordinate step.
INSIDE_TOLERANCE = 1e-7

# The bounds of rejected vertices are halved, within one outer iteration,
# until each is at most its `bounded_burst` retry bound, at most this many
# times: enough to bring any step the optimizer's floor allows below a µm.
# Beyond that they wait for the next one.
MAX_BOUND_HALVINGS = 8

# A retry's bounds need shrink no further than this share of a bubble.
RETRY_SHARE = 16

LBFGSB_ABNORMAL = 2

# Edges shorter than this fraction of the spacing meet the short-edge barrier.
# Resampling leaves every edge at a free vertex at least half the spacing.
SHORT_EDGE_FRACTION = 0.1

# The chain's length term is ill-conditioned (tension spreads one vertex per
# iteration), so L-BFGS-B's default stopping tolerances end a burst long before
# the shape settles; with these defaults, burst_maxiter bounds the burst instead.
LBFGSB_FTOL = 1e-15
LBFGSB_GTOL = 1e-10


@dataclass(frozen=True)
class OptimizerSettings:
    spacing: float = 0.1
    """Target distance between resampled vertices, mm."""
    bubble_tolerance: float = 0.01
    """δ, mm: bubbles are measured to obstacles inflated by δ less than the
    clearance, and vertices deeper than δ inside an obstacle are frozen."""
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
    burst_maxiter: int = 100
    """L-BFGS-B iterations per burst of a problem."""
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


def half_side(distance: FloatArray, settings: OptimizerSettings) -> FloatArray:
    """Half the side of the box inscribed in each bubble: half the free
    distance, δ allowed inside the clearance."""
    return np.maximum(distance + settings.bubble_tolerance, 0.0) / 2 / math.sqrt(2)


def point_distance(field: DistanceField, point: FloatArray) -> float:
    return float(field.signed_distance(point[None])[0][0])


Objective = Callable[[FloatArray], tuple[float, FloatArray]]
Revert = Callable[[FloatArray], tuple[BoolArray, int]]
Crossings = Callable[[FloatArray], tuple[int, BoolArray]]


@dataclass(frozen=True)
class Burst:
    points: FloatArray
    """Where the burst left its flat coordinates."""
    rejected_edges: int
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


def bounded_burst(
    objective: Objective,
    start: FloatArray,
    half_sides: FloatArray,
    revert: Revert,
    settings: OptimizerSettings,
    retry_bounds: FloatArray | None = None,
) -> Burst:
    """Minimise `objective` from `start` with each coordinate within its
    `half_sides` of it, retried with halved bounds where edges crossed.

    `revert` moves back, in place, the coordinates of edges that newly cross
    an obstacle, and returns which it moved and how many edges crossed. A
    run L-BFGS-B ends abnormally is taken as its check leaves it, and so is
    one whose moved-back coordinates all have bounds at most their
    `retry_bounds` once halved, by default a sixteenth of their
    `half_sides`. Where every coordinate's bounds coincide in floating
    point, nothing can move, and the burst ends where it is.
    """
    if retry_bounds is None:
        retry_bounds = half_sides / RETRY_SHARE
    scale = np.ones(len(start))
    current = start.copy()
    rejected_edges = 0
    abnormal = False
    for _ in range(MAX_BOUND_HALVINGS + 1):
        reach = half_sides * scale
        lower, upper = start - reach, start + reach
        if np.array_equal(lower, upper):
            # L-BFGS-B would return a result without a status.
            break
        result = minimize(
            objective,
            current,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper, strict=True)),
            options={
                "maxiter": settings.burst_maxiter,
                "ftol": settings.lbfgs_ftol,
                "gtol": settings.lbfgs_gtol,
            },
        )
        abnormal = result.status == LBFGSB_ABNORMAL
        current = np.array(result.x, dtype=float)
        reverted, crossings = revert(current)
        if crossings == 0 or abnormal:
            break
        rejected_edges += crossings
        scale[reverted] /= 2
        if np.all(half_sides[reverted] * scale[reverted] <= retry_bounds[reverted]):
            break
    return Burst(current, rejected_edges, abnormal)
