"""Obstacle-aware simplification of optimised polylines.

A chord around a convex obstacle always deviates toward it, so merging by
chord tolerance alone would cut into the clearance at every wrap; each merged
segment must also pass the exact edge check.

Several chains are simplified in turn, each against its own field plus the
other chains on its layer at their latest shapes, those it ignores left
out, as the optimizer leaves them out (`neighbours.neighbour_field`). Its
field holds only the chains whose copper can reach the vertices it merges
across, found by comparing boxes with those of all chains on its layer in
one vectorised step (`_Neighbours`). The field it builds and searches so
grows with the chains near it, not with all chains on the layer, and the
result is that of a field of every chain on the layer. A vertex that
keeps a pad or via (`chain.Contact`) is fixed, so merging keeps it where
the optimizer left it, inside its region. A coupled section of a
differential pair is simplified as its centreline, as the optimizer moved
it, and its two tracks follow it as they do in the optimizer.

`kept_vertices` merges a section's routing the same way before it moves
(`pair_blocks`).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from kibends.bursts import DEFAULT_CHORD_TOLERANCE, INSIDE_TOLERANCE
from kibends.energy import BoolArray, FloatArray, IntArray, ObstacleField
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import track_inflation

if TYPE_CHECKING:
    # Both import pair_blocks, which simplifies a section's routing.
    from kibends.pair_blocks import PairBlock
    from kibends.results import RunResult


def simplify(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> FloatArray:
    """Merge consecutive segments while the chord stays within `tolerance` of
    the merged vertices and clear of obstacles; fixed vertices are kept."""
    return polyline[
        kept_vertices(polyline, fixed, field, np.full(len(polyline), tolerance))
    ]


def simplify_chains(
    chains: Sequence[ChainInput],
    run: RunResult,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
    *,
    sections: Mapping[int, PairBlock] | None = None,
) -> list[FloatArray]:
    """Simplify the optimised chains of `run`, each against its field and the
    other chains on its layer that it does not ignore, at their latest
    shapes.

    Each of `sections` stands in for the chain of its index, its polyline
    a coupled section's centreline. A section merges within its block's
    tolerance and keeps its merged edges clear of its field as the block's `facing`,
    which leaves out the chains of its two nets. Its tracks' mitred corners
    at kept vertices next to a merged edge reach farther out than the
    edges: where one enters the field, the vertices on either side of it
    are kept and the section merged again. The others see its two tracks,
    offset from its centreline at its latest shape.
    """
    if len(run.chains) != len(chains):
        raise ValueError(
            f"{len(run.chains)} optimised chains for {len(chains)} chain inputs"
        )
    seen = _Neighbours(chains, [c.polyline for c in run.chains], sections or {})
    for i in range(len(chains)):
        result = run.chains[i]
        # Each merged edge joins two of these vertices, so it stays in their box.
        field = seen.field(i, result.polyline)
        fixed = result.fixed
        allowed = seen.tolerances(i, result.polyline, tolerance)
        while True:
            kept = kept_vertices(result.polyline, fixed, field, allowed)
            loose = seen.loose_corners(i, kept)
            if not loose.any():
                break
            fixed = fixed | loose
        seen.move(i, result.polyline[kept])
    return seen.current


class _Neighbours:
    """The chains as they see each other while simplifying, coupled
    sections among them, each at its latest shape.

    A chain's field holds only the others whose copper can reach what it
    asks about: those whose box, around what they show, comes within the
    widest inflation it keeps from a track on its layer of the box around
    the points asked about. Any other is farther from each of those
    points, and from each segment between them, than the distance it is
    kept, so it neither blocks a segment nor makes a distance negative, and
    the answers the simplification takes are those of a field of every
    chain on the layer.
    Finding them compares the boxes of all chains on the layer in one
    vectorised step; the field built and searched grows only with the
    chains near.
    """

    def __init__(
        self,
        chains: Sequence[ChainInput],
        optimised: Sequence[FloatArray],
        sections: Mapping[int, PairBlock],
    ) -> None:
        self.chains = chains
        self.optimised = optimised
        self.sections = sections
        self.current = list(optimised)
        self._shown = [self.shown(j, points) for j, points in enumerate(optimised)]
        self._low = np.empty((len(chains), 2))
        self._high = np.empty((len(chains), 2))
        for j in range(len(chains)):
            self._bound(j)
        on: dict[str, list[int]] = {}
        for j, chain in enumerate(chains):
            on.setdefault(chain.layer, []).append(j)
        layers: dict[str, IntArray] = {
            layer: np.array(members, dtype=np.intp) for layer, members in on.items()
        }
        kinds = {
            layer: {
                (other.centreline.width, other.centreline.clearance)
                for j in members
                for other, _ in self._shown[j]
            }
            for layer, members in on.items()
        }
        self._layer_mates = [layers[chain.layer] for chain in chains]
        """Per chain, the indices of the chains on its layer, its own among
        them."""
        self._reach: list[float] = []
        """Per chain, the widest inflation it keeps from a track on its layer."""
        for i, chain in enumerate(chains):
            me = self.viewer(i)[0].centreline
            self._reach.append(
                max(
                    track_inflation(me.width, me.clearance, width, clearance)
                    for width, clearance in kinds[chain.layer]
                )
            )

    def viewer(self, i: int) -> tuple[ChainInput, frozenset[str] | None]:
        """The chain i keeps clear of the others as, and the nets it ignores
        beyond `neighbour_field`'s default: a section's two."""
        block = self.sections.get(i)
        return (self.chains[i], None) if block is None else (block.facing, block.nets)

    def field(self, i: int, points: FloatArray) -> ObstacleField:
        """Chain i's field, with the others as they are now, for questions
        about `points` and the segments between them."""
        low, high = points.min(axis=0), points.max(axis=0)
        on, reach = self._layer_mates[i], self._reach[i]
        near = on[
            (on != i)
            & (self._low[on] <= high + reach).all(axis=1)
            & (self._high[on] >= low - reach).all(axis=1)
        ]
        own, ignored = self.viewer(i)
        shown = [part for j in near for part in self._shown[int(j)]]
        return chain_field(
            self.chains[i].field, neighbour_field(own, shown, ignored=ignored)
        )

    def move(self, j: int, points: FloatArray) -> None:
        """Chain j is at `points` from now on."""
        self.current[j] = points
        self._shown[j] = self.shown(j, points)
        self._bound(j)

    def _bound(self, j: int) -> None:
        shown = np.concatenate([points for _, points in self._shown[j]])
        self._low[j] = shown.min(axis=0)
        self._high[j] = shown.max(axis=0)

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

    def loose_corners(self, i: int, kept: Sequence[int]) -> BoolArray:
        """For a section, the vertices of its optimised centreline to keep
        on either side of each kept vertex next to a merged edge where a
        track's corner enters its field."""
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
        corners = [track[1:-1] for track in block.offset(polyline[at])]
        field = self.field(i, np.concatenate(corners))
        for corner in corners:
            distance, _ = field.signed_distance(corner)
            entering = beside & (distance < -inside - INSIDE_TOLERANCE)
            for vertex in at[1:-1][entering]:
                loose[[vertex - 1, vertex + 1]] = True
        return loose

    def shown(self, j: int, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
        """What the others see of chain j at `points`: a section's tracks."""
        block = self.sections.get(j)
        if block is None:
            return [(self.chains[j], points)]
        tracks = block.offset(points)
        return list(zip(block.tracks, tracks, strict=True))


def kept_vertices(
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
