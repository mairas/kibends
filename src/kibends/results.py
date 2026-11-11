"""What the optimizer returns (`optimize.optimize_chains`): a `RunResult`
with a `ChainResult` per chain, a `ViaResult` per moving via and a
`PairResult` per coupled section, each chain's and section's statistics
before the run (`PolylineStats`, from `polylines.measure`) and how the run
ended (`Termination`).
`OptimizeResult` is the single-chain form `optimize.optimize` returns.
`run` turns these into its reports.
"""

from dataclasses import dataclass
from enum import Enum

from kibends.chain import Point2
from kibends.energy import BoolArray, FloatArray, IntArray
from kibends.pair_blocks import PairBlock
from kibends.pairs import BendCheck


class Termination(Enum):
    CONVERGED = "converged"
    NOTHING_TO_DO = "nothing to do"
    ITERATION_CAP = "iteration cap reached"
    TIME_LIMIT = "time limit reached"
    GAVE_UP = "stopped on abnormal bursts"
    """Every problem settled, some of them only after
    `optimize.MAX_ABNORMAL_BURSTS` abnormal bursts in a row, and the run's
    best state is lower in energy than its inputs; or, for the whole stage
    (`run.OptimizerReport`), the last run converged after an earlier one gave
    up. The best state is used, though it may be short of the minimum."""
    ABNORMAL = "L-BFGS-B terminated abnormally"
    """Some problem stopped after `optimize.MAX_ABNORMAL_BURSTS` abnormal
    bursts in a row and the run's best state is no lower in energy than its
    inputs: nothing usable was found."""


@dataclass(frozen=True)
class PolylineStats:
    length: float
    segment_count: int
    tightest_bend_radius: float
    """Smallest discrete radius of curvature after resampling at the spacing;
    infinite for a straight polyline, mm."""


@dataclass(frozen=True)
class OptimizeResult:
    polyline: FloatArray
    fixed: BoolArray
    """Pinned vertices of `polyline`; simplification keeps them."""
    termination: Termination
    iterations: int
    rejected_edges: int
    """Edges whose burst moves were undone because they crossed an obstacle."""
    abnormal_bursts: int
    """Bursts L-BFGS-B ended abnormally; each was judged like any other."""
    before: PolylineStats
    after: PolylineStats


@dataclass(frozen=True)
class ChainResult:
    polyline: FloatArray
    fixed: BoolArray
    """Pinned vertices of `polyline`, and those that keep a contact;
    simplification keeps them."""
    rejected_edges: int
    """Edges whose burst moves were undone because they crossed an obstacle."""
    before: PolylineStats
    contact_of: IntArray
    """Per vertex of `polyline`, the index among its input's contacts of
    the one it keeps, -1 for none."""


@dataclass(frozen=True)
class ViaResult:
    position: Point2
    rejected_edges: int
    """End edges of its chains whose crossing an obstacle undid its move."""


@dataclass(frozen=True)
class PairResult:
    block: PairBlock
    """The section as the run took it."""
    centreline: FloatArray
    fixed: BoolArray
    """The vertices of `centreline` that simplification keeps: those the run
    pinned, but the direction pins of a section it moved, as a
    merged edge stays within the chord tolerance of them."""
    a: FloatArray
    b: FloatArray
    """The two tracks along `centreline`, b running the same way as a; the
    section's own tracks when the run did nothing."""
    rejected_edges: int
    before: PolylineStats
    """Of the starting centreline."""
    bend: BendCheck | None
    """The centreline's bend radius against the limit of the section's
    gap: its routing's for a section held because that bends too
    tightly, its result's for one the run moved; else None."""

    def kept(self) -> tuple[FloatArray, FloatArray]:
        """The tracks the run leaves: the section's own where it is held or
        the run did nothing, else `a` and `b`."""
        if self.block.held or self.bend is None:
            given = self.block.given
            return given.a.points, given.b.points
        return self.a, self.b


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
