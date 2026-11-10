"""The pairwise via term: what a ground via and a regular one, or ground
vias of two ground nets, give each other, summed over every pair in range.

A ground via is on one of the run's ground nets, a regular one on any other
net (`vias.ViaInput.ground`, `vias.StaticVia`). Two vias interact here only
where their layer spans overlap, and each pair is counted from the side of
each moving via in it, weighted by that via's diameter, as the optimizer's
other via terms are; a via standing still has no side of its own.

- A ground via and a regular one: the full obstacle term
  (`energy.obstacle_integrand`), repulsion and clearance hinge, at the
  distance by which their centres are farther apart than they keep, plus
  the attraction (`energy.attraction_integrand`), continued linearly below
  that distance. Together they give each pair a neutral distance.
- Ground vias of two ground nets: the full obstacle term alone.

Summed over pairs rather than taken for the nearest only: against summed
attraction, nearest-only repulsion has no neutral distance once anything
else is nearer. These pairs are therefore left out of the optimizer's
nearest-only via terms and out of a moving via's static field, which then
holds none of the vias standing still that it pairs with
(`obstacles.static_via_obstacles`). Clearance judging is unchanged: the
pairs keep the distances `obstacles.via_via_inflation` gives.

The problem's other vias are paired once, at setup, at the distances the
optimizer gives (`optimize._Via.apart`). The vias standing still, unselected
ones and those the run did not let move, are found near each moving via at
every evaluation with one k-d tree, so the energy is one function of the
positions however far the vias move; the distance a moving via keeps from
one is worked out the first time they meet.
"""

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import chain

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    IntArray,
    attraction_integrand,
    obstacle_integrand,
    obstacle_reach,
)
from kibends.obstacles import via_via_inflation
from kibends.snapshot import Via
from kibends.vias import StaticVia, ViaInput


class PairKind(enum.IntEnum):
    """How two vias interact in the pairwise term."""

    RETURN = 0
    """A ground via and a regular one: attraction and the full obstacle
    term."""
    GROUNDS = 1
    """Ground vias of two ground nets: the full obstacle term."""


def pair_kind(
    via: Via, ground: bool, other: Via, other_ground: bool
) -> PairKind | None:
    """How `via` and `other`, each a ground via or not, interact in the
    pairwise term; None if they do not."""
    if via.net == other.net or not via.shares_layer(other):
        return None
    if ground and other_ground:
        return PairKind.GROUNDS
    if ground or other_ground:
        return PairKind.RETURN
    return None


_NO_PAIR = -1


class ViaPairs:
    """The pairwise term of a problem's moving vias, with the problem's
    other vias and with the vias standing still."""

    def __init__(
        self,
        vias: Sequence[ViaInput],
        free: Sequence[int],
        apart: Sequence[Mapping[int, float]],
        static: Sequence[StaticVia],
        params: EnergyParams,
        *,
        hole_clearance: float,
        hole_to_hole: float,
    ) -> None:
        """The pairs of the moving vias `free`, by their index among the
        problem's `vias`, which keep the distances `apart` gives from the
        problem's other vias, and of them with the `static` vias, which
        keep the distances `via_via_inflation` gives with `hole_clearance`
        and `hole_to_hole`."""
        self.vias = vias
        self.free = np.array(free, dtype=np.intp)
        self.free_diameters = np.array(
            [vias[k].via.diameter for k in free], dtype=float
        )
        self.hole_clearance = hole_clearance
        self.hole_to_hole = hole_to_hole
        self.obstacle = obstacle_integrand(params)
        self.attraction = attraction_integrand(params)
        first: list[int] = []
        second: list[int] = []
        distances: list[float] = []
        kinds: list[PairKind] = []
        for k in free:
            given = vias[k]
            for u, distance in apart[k].items():
                kind = pair_kind(given.via, given.ground, vias[u].via, vias[u].ground)
                if kind is not None:
                    first.append(k)
                    second.append(u)
                    distances.append(distance)
                    kinds.append(kind)
        self.first = np.array(first, dtype=np.intp)
        self.second = np.array(second, dtype=np.intp)
        self.apart = np.array(distances, dtype=float)
        self.kinds = np.array(kinds, dtype=np.int8)
        self.weights = np.array([vias[k].via.diameter for k in first], dtype=float)
        self.paired = [set[int]() for _ in vias]
        """Per via, the problem's vias it pairs with."""
        for k, u in zip(first, second, strict=True):
            self.paired[k].add(u)

        static_kinds = np.array(
            [
                [
                    _NO_PAIR
                    if (kind := pair_kind(vias[k].via, vias[k].ground, s.via, s.ground))
                    is None
                    else kind
                    for s in static
                ]
                for k in free
            ],
            dtype=np.int8,
        ).reshape(len(free), len(static))
        met = (static_kinds != _NO_PAIR).any(axis=0)
        self.static = [s for s, m in zip(static, met, strict=True) if m]
        self.static_kinds = static_kinds[:, met]
        self.static_apart = np.full(self.static_kinds.shape, math.nan)
        """By moving via and static via, the distance their centres keep;
        NaN until they first meet."""
        at = np.array([s.via.position for s in self.static], dtype=float)
        self.tree = cKDTree(at.reshape(-1, 2)) if self.static else None
        self.static_at = at.reshape(-1, 2)
        reach = obstacle_reach(params)
        cutoff = params.attraction_range if params.attraction else 0.0
        self.static_reach = np.array(
            [max(cutoff, self._widest(vias[k]) + reach) for k in free], dtype=float
        )
        """By moving via, how far from its centre a static via may give it
        energy."""

    @property
    def empty(self) -> bool:
        return not len(self.first) and not self.static

    def energy(self, positions: Sequence[FloatArray]) -> tuple[float, FloatArray]:
        """The term's value with the problem's vias at `positions`, and its
        gradient at each via's centre."""
        at = np.array(positions, dtype=float).reshape(-1, 2)
        gradient = np.zeros_like(at)
        value, push = self._terms(
            at[self.first] - at[self.second], self.apart, self.kinds, self.weights
        )
        np.add.at(gradient, self.first, push)
        np.add.at(gradient, self.second, -push)
        if self.tree is not None:
            rows, columns = self._static_near(at[self.free])
            moving = self.free[rows]
            static_value, static_push = self._terms(
                at[moving] - self.static_at[columns],
                self.static_apart[rows, columns],
                self.static_kinds[rows, columns],
                self.free_diameters[rows],
            )
            value += static_value
            np.add.at(gradient, moving, static_push)
        return value, gradient

    def _static_near(self, at: FloatArray) -> tuple[IntArray, IntArray]:
        """The pairs of a moving via, by its row of `at`, and a static via
        it pairs with that is near enough to give it energy, with the
        distance they keep worked out."""
        assert self.tree is not None
        free: list[int] = self.free.tolist()
        found = self.tree.query_ball_point(at, self.static_reach, return_sorted=False)
        counts = np.fromiter(map(len, found), dtype=np.intp, count=len(found))
        rows = np.repeat(np.arange(len(found), dtype=np.intp), counts)
        columns = np.fromiter(
            chain.from_iterable(found), dtype=np.intp, count=int(counts.sum())
        )
        paired = self.static_kinds[rows, columns] != _NO_PAIR
        rows, columns = rows[paired], columns[paired]
        unknown: BoolArray = np.isnan(self.static_apart[rows, columns])
        new_rows: list[int] = rows[unknown].tolist()
        new_columns: list[int] = columns[unknown].tolist()
        for n, s in zip(new_rows, new_columns, strict=True):
            self.static_apart[n, s] = self._kept(free[n], self.static[s])
        return rows, columns

    def _kept(self, k: int, other: StaticVia) -> float:
        """The distance moving via k's centre keeps from `other`'s, as its
        static field would count it: with its zones on the layers `other`
        spans."""
        given = self.vias[k]
        clearances = {given.via.net: given.clearance, other.via.net: other.clearance}
        zones = [z for z in given.zones if any(map(other.via.reaches, z.layers))]
        distance = via_via_inflation(
            given.via,
            other.via,
            clearances.__getitem__,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
            zones=zones,
        )
        # Vias of other nets always keep their copper clearance.
        assert distance is not None
        return distance

    def _widest(self, given: ViaInput) -> float:
        """The largest distance `given`'s centre keeps from a static via it
        pairs with: from one as wide, as widely drilled and of as wide a
        clearance as the widest, of a net other than its own."""
        if not self.static:
            return 0.0
        widest = replace(
            given.via,
            net=f"{given.via.net}\0other",
            diameter=max(s.via.diameter for s in self.static),
            drill=max(s.via.drill for s in self.static),
        )
        clearance = max(given.clearance, *(s.clearance for s in self.static))
        distance = via_via_inflation(
            given.via,
            widest,
            lambda _: clearance,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
            zones=given.zones,
        )
        assert distance is not None
        return distance

    def _terms(
        self,
        delta: FloatArray,
        apart: FloatArray,
        kinds: NDArray[np.int8],
        weights: FloatArray,
    ) -> tuple[float, FloatArray]:
        """The pairs' summed value and each pair's gradient at its first
        via's centre, from the vectors `delta` from its second via's centre
        to its first's, the distances `apart` their centres keep, their
        `kinds` and the first vias' `weights`."""
        if not len(delta):
            return 0.0, np.zeros((0, 2))
        r = np.linalg.norm(delta, axis=1)
        value, slope = self.obstacle(r - apart)
        attracting = kinds == PairKind.RETURN
        if attracting.any():
            pull, pull_slope = self.attraction(r[attracting], apart[attracting])
            value[attracting] += pull
            slope[attracting] += pull_slope
        units = np.divide(
            delta, r[:, None], out=np.zeros_like(delta), where=r[:, None] > 0.0
        )
        return float(weights @ value), (weights * slope)[:, None] * units
