"""The members of a problem as the Rust energy takes them; what they see of
each other and their summed energy are Rust's (`kibends._core.Members`;
their design is in `kibends_core::members` and `kibends_core::seen`).

A `Member` is a chain or a coupled section as its centreline. The others
see a chain as itself and a section as its two tracks (`shown`), laid one
after another, then each via's centre twice (`shown_flat`). `core_members`
reads the members once into the Rust `Members`, which gives what each
member sees of the others (`seen_parts`) and their summed energy
(`members_energy`).
"""

from collections.abc import Mapping, Sequence

import numpy as np

from kibends import _core
from kibends.energy import BoolArray, EnergyParams, FloatArray
from kibends.neighbours import ChainInput
from kibends.obstacles import core_field
from kibends.pair_blocks import PairBlock
from kibends.pairs import diff_pair_nets
from kibends.seen import Seen

Member = ChainInput | PairBlock
"""What a problem optimises: a chain, or a coupled section as its
centreline."""


def shown(member: Member, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
    """What the others see of a member at `points`: a chain itself, a
    section its two tracks."""
    if isinstance(member, PairBlock):
        return list(zip(member.tracks, member.offset(points), strict=True))
    return [(member, points)]


def body_of(member: Member) -> ChainInput:
    """The chain a member's own energy and pins are those of."""
    return member.chain if isinstance(member, PairBlock) else member


def viewer_of(member: Member) -> tuple[ChainInput, frozenset[str]]:
    """The chain a member keeps clear of the others as, and the nets whose
    tracks it ignores, as `neighbours.neighbour_field` takes them: a
    chain's own and the other net of its diff pair, a section's two."""
    if isinstance(member, PairBlock):
        return member.facing, member.nets
    return member, diff_pair_nets(member.centreline.net)


def shown_flat(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray] = (),
) -> FloatArray:
    """What the others see of every member, polylines one after another,
    then each via's centre twice, an edge of no length."""
    return np.concatenate(
        [
            np.empty((0, 2)),
            *(
                points
                for member, line in zip(members, polylines, strict=True)
                for _, points in shown(member, line)
            ),
            *(np.array([at, at]) for at in positions),
        ]
    )


def _trace(chain: ChainInput) -> tuple[str, float, float]:
    line = chain.centreline
    return line.net, line.width, line.clearance


def core_member(member: Member, *, hinged: bool = True) -> _core.Member:
    """`member` as the Rust energy takes it: what it keeps clear of the
    others as and the nets it ignores (`viewer_of`), the tracks it shows
    (`shown`), its body's field, width and contacts (`body_of`), and a
    section's offset and, `hinged`, its bend-radius limit."""
    own, ignored = viewer_of(member)
    body = body_of(member)
    contacts = [core_field(field) for _, field in body.contacts]
    section: _core.PairOffset | None = None
    limit: float | None = None
    tracks = [body]
    if isinstance(member, PairBlock):
        section = member.core
        limit = member.limit if hinged else None
        tracks = list(member.tracks)
    return _core.Member(
        own.layer,
        _trace(own),
        sorted(ignored),
        [_trace(track) for track in tracks],
        core_field(body.field),
        body.centreline.width,
        contacts,
        section,
        limit,
    )


def core_members(members: Sequence[Member]) -> _core.Members:
    """`members` as the Rust energy takes them (`core_member`)."""
    return _core.Members([core_member(member) for member in members])


def _points(polylines: Sequence[FloatArray]) -> list[FloatArray]:
    return [np.asarray(points, dtype=float).reshape(-1, 2) for points in polylines]


def positions_array(positions: Sequence[FloatArray]) -> FloatArray:
    """The vias' centres as one (n, 2) array."""
    return np.array(positions, dtype=float).reshape(-1, 2)


def seen_parts(
    chains: Sequence[Member],
    polylines: Sequence[FloatArray],
    rings: Sequence[Mapping[int, float]] = (),
) -> Seen:
    """For each chain or section at its polyline, the others on its layer at
    theirs as `neighbours.neighbour_field` shows them to it. A section sees
    as its centreline of the diff pair's outer extent, and is seen as its two
    tracks. Each via of `rings` is seen by the members it names, inflated
    by the distance it gives."""
    return core_members(chains).seen(_points(polylines), [dict(r) for r in rings])


def members_energy(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Seen,
    positions: Sequence[FloatArray],
) -> tuple[float, list[FloatArray], FloatArray]:
    """The summed energy of chains and sections on any layers, each in its
    static field plus what it sees of the others on its own layer at
    `polylines` as `seen` shows them, with vias at `positions` among them,
    a section with its bend-radius hinge; its gradient at every vertex of
    every chain and centreline, with the rows of `masks` zeroed; and the
    gradient at each via's centre. Raises ValueError if it is not finite
    (`kibends_core::members::Members::energy`)."""
    return core_members(members).energy(
        _points(polylines),
        [np.asarray(mask, dtype=bool) for mask in masks],
        params,
        short_edge,
        seen,
        positions_array(positions),
    )
