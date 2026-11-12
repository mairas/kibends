"""The pairwise via term: what a ground via and a regular one, or ground
vias of two ground nets, give each other, summed over every pair in range;
the term is Rust's (`kibends._core.ViaPairs`; its design is in
`kibends_core::vias::pairs`).

A ground via is on one of the run's ground nets, a regular one on any other
net (`vias.ViaInput.ground`, `vias.StaticVia`). Two vias interact as
`vias.pair_kind` says, only where their layer spans overlap: a ground via
and a regular one attract to a neutral distance, ground vias of two ground
nets repel with the full obstacle term, and ground vias of one ground net
spread apart by the repulsion alone. The pairs of other nets are left out
of the optimizer's nearest-only via terms and out of a moving via's static
field (`obstacles.split_via_obstacles`); clearance judging is unchanged:
the pairs keep the distances `obstacles.via_via_inflation` gives.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from kibends import _core
from kibends.energy import EnergyParams, FloatArray
from kibends.obstacles import core_via, core_zones
from kibends.vias import StaticVia, ViaInput


class ViaPairs:
    """The pairwise term of a problem's moving vias, with each other and
    with the vias standing still."""

    def __init__(
        self,
        vias: Sequence[ViaInput],
        apart: Sequence[Mapping[int, float]],
        static: Sequence[StaticVia],
        params: EnergyParams,
        *,
        hole_clearance: float,
        hole_to_hole: float,
    ) -> None:
        """The pairs of the problem's moving `vias`, which keep the
        distances `apart` gives from each other, and of them with the
        `static` vias, which keep the distances `via_via_inflation` gives
        with `hole_clearance` and `hole_to_hole`."""
        self.vias = vias
        self.core = _core.ViaPairs(
            [core_via(v.via) for v in vias],
            [v.ground for v in vias],
            [v.clearance for v in vias],
            [core_zones(v.zones) for v in vias],
            [dict(a) for a in apart],
            [core_via(s.via) for s in static],
            [s.ground for s in static],
            [s.clearance for s in static],
            params,
            hole_clearance,
            hole_to_hole,
        )
        self.paired = self.core.paired
        """Per via, the problem's vias it pairs with."""

    def _kept(self, k: int, other: StaticVia) -> float | None:
        """The distance moving via k's centre keeps from `other`'s, as its
        static field would count it."""
        return self.core.kept(k, core_via(other.via), other.clearance)

    @property
    def empty(self) -> bool:
        return self.core.empty

    def energy(self, positions: Sequence[FloatArray]) -> tuple[float, FloatArray]:
        """The term's value with the problem's vias at `positions`, and its
        gradient at each via's centre."""
        return self.core.energy(np.array(positions, dtype=float).reshape(-1, 2))
