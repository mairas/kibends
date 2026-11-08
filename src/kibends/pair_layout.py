"""Differential pairs among the selected chains, and how a run lays them out
for the optimizer.

A pair's coupled sections become blocks of their own (`pair_blocks`), its
uncoupled pieces ordinary chains (`split_pair`).
"""

from collections.abc import Callable
from dataclasses import replace

from kibends.energy import ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.pair_blocks import PairInput, end_gaps
from kibends.pairs import Pair, Piece, Track

SectionField = Callable[[str, ChainCentreline, ChainCentreline, float], ObstacleField]
"""The field of a coupled section's centreline, from its layer, its two
tracks, b running the same way as a, and its gap."""


def split_pair(
    pair: Pair, a: ChainInput, b: ChainInput, field: SectionField
) -> tuple[
    list[PairInput], list[tuple[Piece, ChainInput]], list[tuple[Piece, ChainInput]]
]:
    """The coupled sections of `pair` as pair inputs, each with its `field`,
    and the uncoupled pieces of each of its chains, a's and then b's, as
    chain inputs pinned at both ends.

    `a` and `b` are its chains with their vertices and pins as `pair.a` and
    `pair.b` hold them. A piece of one vertex, where a section reaches a
    chain's end, is left out.
    """
    sections: list[PairInput] = []
    for s in pair.sections:
        points_a, points_b = pair.coupled(s)
        pinned_a = pair.a.pinned[s.a_start : s.a_end + 1]
        pinned_b = pair.b.pinned[s.b_start : s.b_end + 1]
        if s.b_reversed:
            points_b, pinned_b = points_b[::-1].copy(), pinned_b[::-1]
        track_a = replace(a.centreline, points=points_a)
        track_b = replace(b.centreline, points=points_b)
        sections.append(
            PairInput(
                track_a,
                track_b,
                pinned_a,
                pinned_b,
                s.gap,
                a.layer,
                # As wide as its widest, where an end's gap is wider.
                field(
                    a.layer,
                    track_a,
                    track_b,
                    max(s.gap, *end_gaps(points_a, points_b, a.centreline.width)),
                ),
            )
        )
    pieces_a, pieces_b = (
        [
            (piece, _piece_input(chain, track, piece))
            for piece in uncoupled
            if piece.end != piece.start
        ]
        for chain, track, uncoupled in zip(
            (a, b), (pair.a, pair.b), pair.uncoupled(), strict=True
        )
    )
    return sections, pieces_a, pieces_b


def _piece_input(chain: ChainInput, track: Track, piece: Piece) -> ChainInput:
    fixed = track.pinned[piece.start : piece.end + 1].copy()
    fixed[[0, -1]] = True
    return replace(
        chain,
        centreline=replace(
            chain.centreline, points=track.between(piece.start, piece.end)
        ),
        fixed=fixed,
    )
