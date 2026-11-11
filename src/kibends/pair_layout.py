"""Differential pairs among the selected chains, and how a run lays them out
for the optimizer.

Two selected chains on one layer whose nets KiCad pairs by name form a diff
pair where they run coupled (`pairs.find_pair`). Where a stub junction,
a selected group's held chain (`chain.Chain.held`), such as a length-tuning
pattern, or unselected copper cuts one net of a diff pair into several
chains, the track of the other net is cut between two of its coupled
sections, where a section ends opposite an end of a chain of the cut net,
within about a pitch (`cut_parts`), and the parts pair one to one. The
sections then run up to the cut on both tracks. A stretch of either track
between a cut and a section is held as drawn; a fan-out to a pad or via
smooths, and so does a part that pairs with nothing, pinned at both ends.
A section or piece of a held chain is held. The optimizer sees a diff pair
as its coupled sections, each a block of its own (`pair_blocks`), and its
uncoupled pieces, each an ordinary chain; the other selected chains stay
as they are.
No via moves at a paired chain, as a diff pair's vias stay fixed
(`vias.may_move`).
After the run, the pieces and the sections' tracks join back into one
polyline per selected chain, which the commit takes like any other chain.
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import NamedTuple

import numpy as np
from loguru import logger

from kibends.chain import Chain
from kibends.coords import UserFrame
from kibends.energy import BoolArray, FloatArray, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.pair_blocks import Hold, PairInput
from kibends.pairs import (
    Opposite,
    Pair,
    PairError,
    Piece,
    Track,
    between_sections,
    cut_chain,
    find_pair,
    nets_pair,
    reach,
    reached,
    runs_coupled,
    vertex_snap,
)
from kibends.vias import ChainEnd

UNPAIRED = "smoothing them as two chains"


@dataclass(frozen=True)
class ChainPart:
    selected: int
    """Index of the selected chain it is a part of."""
    chain: Chain
    """Its stretch of that chain, the whole chain unless `cut_parts` cut it."""
    cut: tuple[bool, bool] = (False, False)
    """Whether its start and its end are cuts: where `cut_parts` cut it, or
    a chain end that a chain of the other net was cut opposite."""


def cut_parts(chains: Sequence[Chain], chord_tolerance: float) -> list[ChainPart]:
    """The parts of `chains`, chain by chain and in order along each.

    A chain is cut, at a pinned vertex, where a coupled section ends on it
    opposite an end of a chain of its diff pair's other net on its layer
    that the section reaches (`pairs.reach`), if that lies between two of
    the chain's coupled sections, within the reach; any other chain is one
    part. Then a chain of a diff pair that a stub or unselected copper cuts
    pairs part by part with the track beside it (`find_pairs`). The end of
    the diff pair, where its tracks fan out, cuts nothing.
    """
    spans: list[list[tuple[float, float]]] = [[] for _ in chains]
    opposite: list[list[tuple[Opposite, int]]] = [[] for _ in chains]
    tolerance = [0.0 for _ in chains]
    for k, m in combinations(range(len(chains)), 2):
        a, b = chains[k], chains[m]
        if a.layer != b.layer or not nets_pair(a.net, b.net):
            continue
        try:
            pair = find_pair(a, b, chord_tolerance)
        except PairError:
            continue
        if pair is None:
            continue
        for own, other, got in zip((k, m), (m, k), reach(pair), strict=True):
            spans[own] += got.spans
            opposite[own] += [(o, other) for o in got.opposite]
            tolerance[own] = max(tolerance[own], reached(pair))
    cuts: list[list[float]] = [[] for _ in chains]
    cut_ends: set[tuple[int, int]] = set()
    for k in range(len(chains)):
        for o, m in opposite[k]:
            if between_sections(o.at, spans[k], tolerance[k]):
                cuts[k].append(o.at)
                cut_ends.add((m, o.end))
    parts: list[ChainPart] = []
    for k, chain in enumerate(chains):
        own = cut_chain(chain, cuts[k], vertex_snap(chord_tolerance))
        for n, part in enumerate(own):
            cut = (n > 0 or (k, 0) in cut_ends, n < len(own) - 1 or (k, -1) in cut_ends)
            parts.append(ChainPart(k, part, cut))
    return parts


@dataclass(frozen=True)
class ChainPair:
    a: int
    b: int
    """Indices of the diff pair's two chains among those `find_pairs` took:
    the parts of the selected chains, in a run."""
    pair: Pair


def find_pairs(
    chains: Sequence[Chain], chord_tolerance: float, *, frame: UserFrame
) -> list[ChainPair]:
    """The differential pairs among `chains`, each chain in at most one.

    Warns about two chains whose nets pair and run coupled but that cannot
    be smoothed as a pair, and about a chain that runs coupled with a chain
    its net pairs with but that pairs with another chain of its net, at its
    start in `frame`; both are smoothed as chains.
    """
    found, taken, missed, skipped, uncoupled = _pair_up(chains, chord_tolerance)
    for k, m, message in missed:
        if not {k, m} & taken:
            logger.warning(message)
    for k, m in uncoupled:
        if not {k, m} & taken:
            a, b = chains[k], chains[m]
            logger.debug(
                f"{a.net} and {b.net} on {a.layer} form a differential pair "
                f"by name but do not run coupled; {UNPAIRED}"
            )
    _warn_partners_taken(chains, skipped, taken, chord_tolerance, frame)
    return found


class _Pairing(NamedTuple):
    found: list[ChainPair]
    taken: set[int]
    missed: list[tuple[int, int, str]]
    """Two chains whose nets pair that cannot pair, and why."""
    skipped: list[tuple[int, int]]
    """Two chains whose nets pair, one of which paired before."""
    uncoupled: list[tuple[int, int]]
    """Two chains whose nets pair that do not run coupled."""


def _pair_up(chains: Sequence[Chain], chord_tolerance: float) -> _Pairing:
    found: list[ChainPair] = []
    taken: set[int] = set()
    missed: list[tuple[int, int, str]] = []
    skipped: list[tuple[int, int]] = []
    uncoupled: list[tuple[int, int]] = []
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
            uncoupled.append((k, m))
            continue
        found.append(ChainPair(k, m, pair))
        taken |= {k, m}
    return _Pairing(found, taken, missed, skipped, uncoupled)


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
    pair: Pair,
    a: ChainInput,
    b: ChainInput,
    field: SectionField,
    cut: tuple[tuple[bool, bool], tuple[bool, bool]] = ((False, False), (False, False)),
) -> tuple[
    list[PairInput], list[tuple[Piece, ChainInput]], list[tuple[Piece, ChainInput]]
]:
    """The coupled sections of `pair` as pair inputs, each with its `field`,
    and the uncoupled pieces of each of its chains, a's and then b's, as
    chain inputs (`_piece_input`), given which ends of each chain are `cut`
    (`ChainPart.cut`).

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
                field(a.layer, track_a, track_b, s.gap),
                Hold.GROUP if a.held or b.held else None,
            )
        )
    pieces_a, pieces_b = (
        [
            (piece, _piece_input(chain, track, piece, ends))
            for piece in uncoupled
            if piece.end != piece.start
        ]
        for chain, track, uncoupled, ends in zip(
            (a, b), (pair.a, pair.b), pair.uncoupled(), cut, strict=True
        )
    )
    return sections, pieces_a, pieces_b


def _piece_input(
    chain: ChainInput, track: Track, piece: Piece, cut: tuple[bool, bool]
) -> ChainInput:
    """An uncoupled piece of `chain`, whose points and pins `track` holds,
    pinned at both ends, as a chain input. A piece of a held chain, or one
    that reaches a `cut` end of its chain, between a section and a stub, is
    held as drawn; a fan-out to a pad or via smooths."""
    fixed = track.pinned[piece.start : piece.end + 1].copy()
    held = (
        chain.held
        or (piece.after is None and cut[0])
        or (piece.before is None and cut[1])
    )
    fixed[[0, -1]] = True
    fixed |= held
    return replace(
        chain,
        centreline=replace(
            chain.centreline, points=track.between(piece.start, piece.end)
        ),
        fixed=fixed,
        held=held,
    )


@dataclass(frozen=True)
class _Stretch:
    """A stretch of a part of a selected chain, by vertex indices into its
    pair track: an uncoupled piece, an optimizer chain, or a track of a
    section."""

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
    """The selected chains laid out whole, in selection order, then the
    uncoupled pieces of each diff pair, then the unpaired parts of cut
    chains."""
    sections: list[PairInput]
    slots: list[int | None]
    """For each selected chain, its index among `inputs`, or None if paired
    or cut."""
    stretches: dict[int, list[_Stretch]]
    """For each part that is paired or of a cut chain, by its index among
    the parts, its stretches in order along it."""
    parts: list[list[int]]
    """For each selected chain, the indices of its parts, in order along it."""
    pair_sections: list[range]
    """For each pair of parts, the indices of its sections."""
    diff_pairs: list[list[int]]
    """For each diff pair, the indices of its pairs of parts: those whose
    selected chains pair with each other, directly or through another."""

    def whole_polylines(
        self,
        chains: Sequence[FloatArray],
        tracks: Sequence[tuple[FloatArray, FloatArray]],
    ) -> list[FloatArray]:
        """One polyline per selected chain, from `chains`, a polyline per
        optimizer input, and `tracks`, the two tracks of each section, b
        running the same way as a."""
        return [
            chains[slot]
            if slot is not None
            else _joined(self.chain_stretches(k), chains, tracks)
            for k, slot in enumerate(self.slots)
        ]

    def part_polyline(
        self,
        part: int,
        chains: Sequence[FloatArray],
        tracks: Sequence[tuple[FloatArray, FloatArray]],
    ) -> FloatArray:
        """The polyline of a paired part, by its index, from `chains` and
        `tracks` as `whole_polylines` takes them."""
        return _joined(self.stretches[part], chains, tracks)

    def chain_stretches(self, chain: int) -> list[_Stretch]:
        """The stretches of a selected chain not laid out whole, in order."""
        return [s for part in self.parts[chain] for s in self.stretches[part]]

    def ends(self, ends: Sequence[ChainEnd]) -> tuple[ChainEnd, ...]:
        """Chain ends of the selected chains, none of them paired, as ends
        of the optimizer's."""
        moved: list[ChainEnd] = []
        for end in ends:
            slot = self.slots[end.chain]
            assert slot is not None
            moved.append(end._replace(chain=slot))
        return tuple(moved)


def _joined(
    stretches: Sequence[_Stretch],
    chains: Sequence[FloatArray],
    tracks: Sequence[tuple[FloatArray, FloatArray]],
) -> FloatArray:
    lines: list[FloatArray] = []
    for stretch in stretches:
        line = _line(stretch, chains, tracks)
        lines.append(line[1:] if lines else line)
    return np.concatenate(lines)


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
    whole: Sequence[ChainInput],
    parts: Sequence[ChainPart],
    pairs: Sequence[ChainPair],
    field: SectionField,
) -> Layout:
    """The optimizer's chains and coupled sections for the selected chains,
    `whole` their inputs, `parts` their parts (`cut_parts`) and `pairs` the
    diff pairs among those, with `field` building each section's field.

    A selected chain of one part that does not pair is laid out whole; a
    part of a cut chain that does not pair is an uncoupled piece pinned at
    both ends."""
    paired = {k for p in pairs for k in (p.a, p.b)}
    of_chain: list[list[int]] = [[] for _ in whole]
    for i, part in enumerate(parts):
        of_chain[part.selected].append(i)
    kept = [
        k for k, own in enumerate(of_chain) if len(own) == 1 and own[0] not in paired
    ]
    inputs = [whole[k] for k in kept]
    slots: list[int | None] = [None] * len(whole)
    for slot, k in enumerate(kept):
        slots[k] = slot

    def part_input(i: int, points: FloatArray, fixed: BoolArray) -> ChainInput:
        given = whole[parts[i].selected]
        return replace(
            given,
            centreline=replace(given.centreline, points=points),
            fixed=fixed,
            contacts=(),
        )

    sections: list[PairInput] = []
    stretches: dict[int, list[_Stretch]] = {}
    pair_sections: list[range] = []
    for p in pairs:
        pair = p.pair
        a, b = (
            part_input(i, track.points, track.pinned)
            for i, track in ((p.a, pair.a), (p.b, pair.b))
        )
        first = len(sections)
        cut = (parts[p.a].cut, parts[p.b].cut)
        new_sections, pieces_a, pieces_b = split_pair(pair, a, b, field, cut)
        sections += new_sections
        pair_sections.append(range(first, len(sections)))
        for i, pieces, on_b in ((p.a, pieces_a, False), (p.b, pieces_b, True)):
            own = [
                _Stretch(piece.start, piece.end, piece=len(inputs) + n)
                for n, (piece, _) in enumerate(pieces)
            ]
            inputs += [given for _, given in pieces]
            for n, s in enumerate(pair.sections):
                start, end = (s.b_start, s.b_end) if on_b else (s.a_start, s.a_end)
                own.append(
                    _Stretch(
                        start,
                        end,
                        section=first + n,
                        b=on_b,
                        reverse=on_b and s.b_reversed,
                    )
                )
            stretches[i] = sorted(own, key=lambda stretch: stretch.start)
    for i, part in enumerate(parts):
        if i in paired or slots[part.selected] is not None:
            continue
        fixed = np.array(part.chain.pinned)
        fixed[[0, -1]] = True
        stretches[i] = [_Stretch(0, len(fixed) - 1, piece=len(inputs))]
        inputs.append(part_input(i, np.array(part.chain.points, dtype=float), fixed))
    inputs = _pinned_where_fixed(inputs, sections)
    joined = [(parts[p.a].selected, parts[p.b].selected) for p in pairs]
    diff_pairs = _diff_pairs(joined)
    return Layout(
        inputs, sections, slots, stretches, of_chain, pair_sections, diff_pairs
    )


def _diff_pairs(pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """The indices of `pairs` of selected chains, grouped by the chains they
    join, each group in order and the groups by their first."""
    groups: list[tuple[set[int], list[int]]] = []
    for i, ends in enumerate(pairs):
        joined = [g for g in groups if g[0] & set(ends)]
        chains = set(ends).union(*(g[0] for g in joined))
        members = sorted([i, *(m for g in joined for m in g[1])])
        groups = [g for g in groups if g not in joined] + [(chains, members)]
    return sorted((members for _, members in groups), key=min)


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
