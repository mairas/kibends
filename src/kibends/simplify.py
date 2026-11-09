"""Obstacle-aware simplification of optimised polylines.

A chord around a convex obstacle always deviates toward it, so merging by
chord tolerance alone would cut into the clearance at every wrap; each merged
segment must also pass the exact edge check.

Several chains are simplified in turn, each against its own field plus the
other chains on its layer at their latest shapes, those of its net left
out, as the optimizer leaves them out. A vertex that keeps a pad or via
(`chain.Contact`) is fixed, so merging keeps it where the optimizer left
it, inside its region. A coupled section of a
differential pair is simplified as its centreline, as the optimizer moved
it, and its two tracks follow it as they do in the optimizer.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from loguru import logger

from kibends.bursts import INSIDE_TOLERANCE
from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    ObstacleField,
    obstacle_reach,
)
from kibends.neighbours import (
    ChainInput,
    chain_field,
    neighbour_field,
    same_layer_neighbours,
)
from kibends.obstacles import CompositeField
from kibends.optimize import RunResult
from kibends.pair_blocks import PairBlock

DEFAULT_CHORD_TOLERANCE = 0.01
"""mm: well below fabrication tolerances; at 5 µm a 23 mm chain on a real
board came out as 63 tracks, and at 0.025 mm the committed tracks strayed
visibly from the optimised curve."""


def simplify(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> FloatArray:
    """Merge consecutive segments while the chord stays within `tolerance` of
    the merged vertices and clear of obstacles; fixed vertices are kept."""
    return polyline[_kept(polyline, fixed, field, np.full(len(polyline), tolerance))]


def simplify_chains(
    chains: Sequence[ChainInput],
    run: RunResult,
    params: EnergyParams,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
    *,
    sections: Mapping[int, PairBlock] | None = None,
) -> list[FloatArray]:
    """Simplify the optimised chains of `run`, each against its field and the
    other chains of other nets on its layer at their latest shapes.

    Each of `sections` stands in for the chain of its index, its polyline
    a coupled section's centreline; sections go first, then the chains in
    the order of `chains`. A section merges within its block's
    tolerance, never across the vertices that bound its ends' gap tapers,
    and keeps its merged edges clear of its field as the block's `facing`,
    with its own nets' chains left out near the fan-outs it touches. Its
    tracks' mitred corners at kept vertices next to a merged edge reach
    farther out than the edges: where one enters the field, the vertices
    on either side of it are kept and the section merged again. The others
    see its two tracks, offset from its centreline at its latest shape with
    each kept vertex at its optimised gap.

    Since the pairwise clearance is symmetric, every merged edge keeps
    clear of the final neighbours; a chain whose merged edges do not is
    logged as an error and left as optimised.

    A section's own nets' chains are trimmed near the fan-outs it touches
    by the reach of `params`, measured against its starting centreline,
    where the optimizer measures against its current one. That is
    deliberate: trimming against the optimised polyline left more segments
    at acute junctions when tried
    (https://github.com/mairas/kibends/issues/17).
    """
    if len(run.chains) != len(chains):
        raise ValueError(
            f"{len(run.chains)} optimised chains for {len(chains)} chain inputs"
        )
    optimised = [c.polyline for c in run.chains]
    seen = _Neighbours(chains, optimised, sections or {}, obstacle_reach(params))
    neighbours = same_layer_neighbours(chains)
    current = list(optimised)
    kept = [list(range(len(polyline))) for polyline in optimised]
    # Sections first: a section leaves its fan-outs out of its field near
    # where they touch it, so they must keep clear of its final tracks.
    order = [*seen.sections, *(i for i in range(len(chains)) if i not in seen.sections)]
    for i in order:
        result = run.chains[i]
        field = chain_field(chains[i].field, seen.fields(i, neighbours[i], current))
        fixed = result.fixed | seen.bounds(i)
        allowed = seen.tolerances(i, result.polyline, tolerance)
        while True:
            kept[i] = _kept(result.polyline, fixed, field, allowed)
            loose = seen.loose_corners(i, kept[i], field)
            if not loose.any():
                break
            fixed = fixed | loose
        current[i] = result.polyline[kept[i]]
    cutting = cutting_chains(
        chains, optimised, current, kept, seen.reach, neighbours, sections=sections
    )
    for i in cutting:
        logger.error(
            f"Chain {chains[i].centreline.description} left unsimplified at"
            f" {len(optimised[i]) - 1} tracks: its simplified edges cut into"
            " a neighbour's clearance"
        )
        current[i] = optimised[i]
    return current


def cutting_chains(
    chains: Sequence[ChainInput],
    optimised: Sequence[FloatArray],
    simplified: Sequence[FloatArray],
    kept: Sequence[Sequence[int]],
    reach: float,
    neighbours: Sequence[Sequence[int]] | None = None,
    *,
    sections: Mapping[int, PairBlock] | None = None,
) -> list[int]:
    """Chains whose merged edges, `optimised[i][kept[i]]` spanning more than
    one optimised edge, enter a neighbour at its `simplified` shape; by
    default every other chain on its layer is its neighbour. `sections` are
    as `simplify_chains` takes them."""
    seen = _Neighbours(chains, optimised, sections or {}, reach)
    cutting: list[int] = []
    if neighbours is None:
        neighbours = same_layer_neighbours(chains)
    for i, indices in enumerate(np.asarray(k) for k in kept):
        merged = np.diff(indices) > 1
        if not (merged.any() and neighbours[i]):
            continue
        starts = optimised[i][indices[:-1][merged]]
        ends = optimised[i][indices[1:][merged]]
        field = CompositeField(seen.fields(i, neighbours[i], simplified))
        if not field.segments_clear(starts, ends).all():
            cutting.append(i)
    return cutting


class _Neighbours:
    """The chains as they see each other while simplifying, coupled
    sections among them."""

    def __init__(
        self,
        chains: Sequence[ChainInput],
        optimised: Sequence[FloatArray],
        sections: Mapping[int, PairBlock],
        reach: float,
    ) -> None:
        self.chains = chains
        self.optimised = optimised
        self.sections = sections
        self.reach = reach

    def tolerances(self, i: int, points: FloatArray, tolerance: float) -> FloatArray:
        """How far each vertex of chain i at `points` may lie off a chord
        merged across it, mm.

        A section's direction pins may be merged across only where they lie
        on the chord, so that its end edges keep the ends' directions and
        both tracks stay parallel there to the edges they meet.
        """
        block = self.sections.get(i)
        if block is None:
            return np.full(len(points), tolerance)
        return np.where(block.direction_pins(points), INSIDE_TOLERANCE, tolerance)

    def loose_corners(
        self, i: int, kept: Sequence[int], field: ObstacleField
    ) -> BoolArray:
        """For a section, the vertices of its optimised centreline to keep
        on either side of each kept vertex next to a merged edge where a
        track's corner enters `field`."""
        polyline = self.optimised[i]
        loose = np.zeros(len(polyline), dtype=bool)
        block = self.sections.get(i)
        if block is None or block.held:
            return loose
        at = np.asarray(kept)
        merged = np.diff(at) > 1
        beside = merged[:-1] | merged[1:]
        if not beside.any():
            return loose
        # Each corner keeps the clearance of a track, half the pair's
        # extent less half a track width inside the centreline's.
        inside = (block.facing.centreline.width - block.given.a.width) / 2
        for track in block.offset_kept(polyline, polyline[at]):
            distance, _ = field.signed_distance(track[1:-1])
            entering = beside & (distance < -inside - INSIDE_TOLERANCE)
            for vertex in at[1:-1][entering]:
                loose[[vertex - 1, vertex + 1]] = True
        return loose

    def bounds(self, i: int) -> BoolArray:
        """The vertices of chain i that merging must keep beyond its fixed
        ones: a section's taper bounds."""
        block = self.sections.get(i)
        if block is None:
            return np.zeros(len(self.optimised[i]), dtype=bool)
        return block.taper_bounds(self.optimised[i])

    def shown(self, j: int, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
        """What the others see of chain j at `points`: a section's tracks."""
        block = self.sections.get(j)
        if block is None:
            return [(self.chains[j], points)]
        tracks = block.offset_kept(self.optimised[j], points)
        return list(zip(block.tracks, tracks, strict=True))

    def fields(
        self,
        i: int,
        others: Sequence[int],
        polylines: Sequence[FloatArray],
    ) -> list[ObstacleField]:
        """The chains `others`, at `polylines`, as obstacles to chain i, a
        section's own nets' chains left out near its fan-outs as measured
        against its starting centreline; those left out entirely are
        dropped."""
        block = self.sections.get(i)
        own = self.chains[i] if block is None else block.facing
        as_net = None if block is None else block.as_net
        return [
            field
            for j in others
            if (
                field := neighbour_field(
                    own,
                    self.shown(j, polylines[j]),
                    self.reach,
                    as_net=as_net,
                )
            )
            is not None
        ]


def _kept(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    tolerances: FloatArray,
) -> list[int]:
    """The vertices that merging keeps, each merged vertex within its
    `tolerances` of the chord."""
    kept = [0]
    last = len(polyline) - 1
    start = 0
    while start < last:
        end = start + 1
        while (
            end < last
            and not fixed[end]
            and _mergeable(polyline, start, end + 1, field, tolerances)
        ):
            end += 1
        kept.append(end)
        start = end
    return kept


def _mergeable(
    polyline: FloatArray,
    start: int,
    end: int,
    field: ObstacleField,
    tolerances: FloatArray,
) -> bool:
    a: FloatArray = polyline[start]
    b: FloatArray = polyline[end]
    chord = b - a
    squared = max(float(chord @ chord), np.finfo(float).tiny)
    inner = polyline[start + 1 : end]
    t = np.clip((inner - a) @ chord / squared, 0.0, 1.0)
    deviation = np.linalg.norm(inner - (a + t[:, None] * chord), axis=1)
    if np.any(deviation > tolerances[start + 1 : end]):
        return False
    return bool(field.segments_clear(a[None], b[None])[0])
