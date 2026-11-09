"""Differential pairs among the selected chains, and how a run lays them out
for the optimizer.

Two selected chains on one layer whose nets KiCad pairs by name form a pair
where they run coupled (`pairs.find_pair`). The optimizer sees a pair as its
coupled sections, each a block of its own (`pair_blocks`), and its uncoupled
pieces, each an ordinary chain; the other selected chains stay as they are.
After the run, the pieces and the sections' tracks join back into one
polyline per selected chain, which the DRC check and the commit take like
any other chain.
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import NamedTuple

import numpy as np
from loguru import logger

from kibends.chain import Chain
from kibends.coords import UserFrame
from kibends.energy import FloatArray, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.pair_blocks import PairInput, end_gaps
from kibends.pairs import (
    Pair,
    PairError,
    Piece,
    Track,
    find_pair,
    nets_pair,
    runs_coupled,
)
from kibends.vias import ChainEnd

UNPAIRED = "smoothing them as two chains"


@dataclass(frozen=True)
class ChainPair:
    a: int
    b: int
    """Indices of the pair's two chains among the selected ones."""
    pair: Pair


def find_pairs(
    chains: Sequence[Chain], chord_tolerance: float, *, frame: UserFrame
) -> list[ChainPair]:
    """The differential pairs among `chains`, each chain in at most one.

    Warns about two chains whose nets pair but that cannot be smoothed as a
    pair, and about a chain that runs coupled with a chain its net pairs
    with but that pairs with another chain of its net, at its start in
    `frame`; both are smoothed as chains.
    """
    found, taken, missed, skipped = _pair_up(chains, chord_tolerance)
    for k, m, message in missed:
        if not {k, m} & taken:
            logger.warning(message)
    _warn_partners_taken(chains, skipped, taken, chord_tolerance, frame)
    return found


def pairs_among(chains: Sequence[Chain], chord_tolerance: float) -> list[ChainPair]:
    """The pairs `find_pairs` finds, without its warnings."""
    return _pair_up(chains, chord_tolerance).found


class _Pairing(NamedTuple):
    found: list[ChainPair]
    taken: set[int]
    missed: list[tuple[int, int, str]]
    """Two chains whose nets pair that cannot pair, and why."""
    skipped: list[tuple[int, int]]
    """Two chains whose nets pair, one of which paired before."""


def _pair_up(chains: Sequence[Chain], chord_tolerance: float) -> _Pairing:
    found: list[ChainPair] = []
    taken: set[int] = set()
    missed: list[tuple[int, int, str]] = []
    skipped: list[tuple[int, int]] = []
    for k, m in combinations(range(len(chains)), 2):
        a, b = chains[k], chains[m]
        if a.layer != b.layer or not nets_pair(a.net, b.net):
            continue
        if {k, m} & taken:
            skipped.append((k, m))
            continue
        try:
            pair = find_pair(a, b, chord_tolerance)
        except PairError as e:
            missed.append((k, m, f"{e}; {UNPAIRED}"))
            continue
        if pair is None:
            missed.append(
                (
                    k,
                    m,
                    f"{a.net} and {b.net} on {a.layer} form a differential pair "
                    f"by name but do not run coupled; {UNPAIRED}",
                )
            )
            continue
        found.append(ChainPair(k, m, pair))
        taken |= {k, m}
    return _Pairing(found, taken, missed, skipped)


def _warn_partners_taken(
    chains: Sequence[Chain],
    skipped: Sequence[tuple[int, int]],
    taken: Collection[int],
    chord_tolerance: float,
    frame: UserFrame,
) -> None:
    """Warn about each chain of `skipped` left unpaired that runs coupled
    with a chain of its pair net that pairs with another chain."""
    warned: set[int] = set()
    for k, m in skipped:
        if (k in taken) == (m in taken):
            continue
        alone, partner = (m, k) if k in taken else (k, m)
        a, b = chains[alone], chains[partner]
        if alone in warned or not runs_coupled(a, b, chord_tolerance):
            continue
        warned.add(alone)
        logger.warning(
            f"{a.net} on {a.layer} from {frame.point(a.points[0])} runs coupled with "
            f"{b.net}, which pairs with another chain of {a.net}; smoothing it "
            "on its own"
        )


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


@dataclass(frozen=True)
class _Stretch:
    """A stretch of a paired chain, by vertex indices into its pair track:
    an uncoupled piece, an optimizer chain, or a track of a section."""

    start: int
    end: int
    piece: int | None = None
    """Index among the optimizer's chains."""
    section: int | None = None
    """Index among the coupled sections."""
    b: bool = False
    """It is the section's track b."""
    reverse: bool = False
    """Track b runs against track a there, so the section holds it reversed."""


@dataclass(frozen=True)
class Layout:
    """The selected chains as the optimizer takes them."""

    inputs: list[ChainInput]
    """The unpaired chains in selection order, then the uncoupled pieces of
    each pair."""
    sections: list[PairInput]
    slots: list[int | None]
    """For each selected chain, its index among `inputs`, or None if paired."""
    stretches: dict[int, list[_Stretch]]
    """For each paired chain, its stretches in order along it."""
    pair_sections: list[range]
    """For each pair, the indices of its sections."""

    def whole_polylines(
        self,
        chains: Sequence[FloatArray],
        tracks: Sequence[tuple[FloatArray, FloatArray]],
    ) -> list[FloatArray]:
        """One polyline per selected chain, from `chains`, a polyline per
        optimizer input, and `tracks`, the two tracks of each section, b
        running the same way as a."""
        whole: list[FloatArray] = []
        for k, slot in enumerate(self.slots):
            if slot is not None:
                whole.append(chains[slot])
                continue
            parts: list[FloatArray] = []
            for stretch in self.stretches[k]:
                line = _line(stretch, chains, tracks)
                parts.append(line[1:] if parts else line)
            whole.append(np.concatenate(parts))
        return whole

    def ends(self, ends: Sequence[ChainEnd]) -> tuple[ChainEnd, ...]:
        """Chain ends of the selected chains as ends of the optimizer's."""
        moved: list[ChainEnd] = []
        for end in ends:
            slot = self.slots[end.chain]
            if slot is None:
                raise ValueError("a moving via joins a chain of a differential pair")
            moved.append(end._replace(chain=slot))
        return tuple(moved)


def _line(
    stretch: _Stretch,
    chains: Sequence[FloatArray],
    tracks: Sequence[tuple[FloatArray, FloatArray]],
) -> FloatArray:
    if stretch.piece is not None:
        return chains[stretch.piece]
    assert stretch.section is not None
    a, b = tracks[stretch.section]
    if not stretch.b:
        return a
    return b[::-1] if stretch.reverse else b


def lay_out(
    whole: Sequence[ChainInput], pairs: Sequence[ChainPair], field: SectionField
) -> Layout:
    """The optimizer's chains and coupled sections for the selected chains,
    `whole` their inputs, with `field` building each section's field."""
    paired = {k for p in pairs for k in (p.a, p.b)}
    unpaired = [k for k in range(len(whole)) if k not in paired]
    inputs = [whole[k] for k in unpaired]
    slots: list[int | None] = [None] * len(whole)
    for slot, k in enumerate(unpaired):
        slots[k] = slot
    sections: list[PairInput] = []
    by_chain: dict[int, list[_Stretch]] = {}
    pair_sections: list[range] = []
    for p in pairs:
        pair = p.pair
        a, b = (
            replace(
                whole[k],
                centreline=replace(whole[k].centreline, points=track.points),
                fixed=track.pinned,
            )
            for k, track in ((p.a, pair.a), (p.b, pair.b))
        )
        first = len(sections)
        new_sections, pieces_a, pieces_b = split_pair(pair, a, b, field)
        sections += new_sections
        pair_sections.append(range(first, len(sections)))
        for k, pieces, on_b in ((p.a, pieces_a, False), (p.b, pieces_b, True)):
            stretches = [
                _Stretch(piece.start, piece.end, piece=len(inputs) + n)
                for n, (piece, _) in enumerate(pieces)
            ]
            inputs += [given for _, given in pieces]
            for n, s in enumerate(pair.sections):
                start, end = (s.b_start, s.b_end) if on_b else (s.a_start, s.a_end)
                stretches.append(
                    _Stretch(
                        start,
                        end,
                        section=first + n,
                        b=on_b,
                        reverse=on_b and s.b_reversed,
                    )
                )
            by_chain[k] = sorted(stretches, key=lambda stretch: stretch.start)
    inputs = _pinned_where_fixed(inputs, sections)
    return Layout(inputs, sections, slots, by_chain, pair_sections)


def _pinned_where_fixed(
    inputs: Sequence[ChainInput], sections: Sequence[PairInput]
) -> list[ChainInput]:
    """`inputs` with every loose end pinned that meets a fixed one, of an
    uncoupled piece or a section's track, as same-net copper pins a
    junction: a junction where a pair starts would otherwise keep some of
    its branches and let the others move off."""
    fixed = {
        (chain.layer, chain.centreline.net, float(x), float(y))
        for chain in inputs
        for end in (0, -1)
        if chain.fixed[end]
        for x, y in [chain.centreline.points[end]]
    } | {
        (section.layer, track.net, float(x), float(y))
        for section in sections
        for track in (section.a, section.b)
        for x, y in track.points[[0, -1]]
    }
    pinned: list[ChainInput] = []
    for chain in inputs:
        ends = chain.fixed.copy()
        for end in (0, -1):
            x, y = chain.centreline.points[end]
            key = (chain.layer, chain.centreline.net, float(x), float(y))
            ends[end] |= key in fixed
        pinned.append(
            chain if np.array_equal(ends, chain.fixed) else replace(chain, fixed=ends)
        )
    return pinned
