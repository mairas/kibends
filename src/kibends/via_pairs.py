"""The pairwise via term: what a ground via and a regular one, or ground
vias of two ground nets, give each other, summed over every pair in range.

A ground via is on one of the run's ground nets, a regular one on any other
net (`vias.ViaInput.ground`, `vias.StaticVia`). Two vias interact here as
`vias.pair_kind` says, only where their layer spans overlap, and each pair
is counted from the side of each moving via in it, weighted by that via's
diameter, as the optimizer's other via terms are; a via standing still has
no side of its own.

- A ground via and a regular one: the full obstacle term
  (`energy.obstacle_integrand`), repulsion and clearance hinge, at the
  distance by which their centres are farther apart than they keep, plus
  the attraction (`energy.attraction_integrand`), continued linearly below
  that distance. Together they give each pair a neutral distance.
- Ground vias of two ground nets: the full obstacle term alone.
- Ground vias of one ground net: the repulsion alone
  (`energy.repulsion_integrand`), beyond their rings and the netclass
  clearance, so that stitching vias drawn toward a signal via spread around
  it. It has no clearance hinge: their only limit stays hole-to-hole, which
  the optimizer's stops keep, so it pushes apart without becoming a wall
  that freezes or blocks them.

Summed over pairs rather than taken for the nearest only: against summed
attraction, nearest-only repulsion has no neutral distance once anything
else is nearer. The pairs of other nets are therefore left out of the optimizer's
nearest-only via terms and out of a moving via's static field, which then
holds none of the vias standing still that it pairs with
(`obstacles.split_via_obstacles`). Clearance judging is unchanged: the
pairs keep the distances `obstacles.via_via_inflation` gives.

Two k-d trees find the pairs near enough to give energy. The problem's
vias are paired once, at setup, at the distances the optimizer gives
(`optimize._Via.apart`), and a tree over their positions, built at every
evaluation, keeps the pairs within reach of each other, so that a grid of
many vias costs as many terms as its vias have neighbours. The vias
standing still, unselected ones and those the run did not let move, sit in
a tree built once, at setup, which finds those near each moving via at
every evaluation, so the energy is one function of the positions however
far the vias move; the distance a moving via keeps from one is worked out
the first time they meet.
"""

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
    repulsion_integrand,
)
from kibends.obstacles import via_via_inflation
from kibends.snapshot import Via
from kibends.vias import PairKind, StaticVia, ViaInput, pair_kind

_NO_PAIR = -1


def _scatter(gradient: FloatArray, index: IntArray, values: FloatArray) -> None:
    """Add each row of `values` to the row of `gradient` that `index`
    gives, as `np.add.at` does, but faster."""
    for axis in range(gradient.shape[1]):
        gradient[:, axis] += np.bincount(
            index, weights=values[:, axis], minlength=len(gradient)
        )


def _spread(given: ViaInput, other: Via, clearance: float) -> float:
    """The centre distance beyond which `given` and `other`, ground vias of
    one net, repel each other: their rings and the larger of their
    clearances, `clearance` being `other`'s."""
    rings = (given.via.diameter + other.diameter) / 2
    return rings + max(given.clearance, clearance)


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
        self.diameters = np.array([v.via.diameter for v in vias], dtype=float)
        self.hole_clearance = hole_clearance
        self.hole_to_hole = hole_to_hole
        self.obstacle = obstacle_integrand(params)
        self.repulsion = repulsion_integrand(params)
        self.attraction = attraction_integrand(params)
        first: list[int] = []
        second: list[int] = []
        distances: list[float] = []
        kinds: list[PairKind] = []
        for k, given in enumerate(vias):
            for u, other in enumerate(vias):
                kind = pair_kind(given.via, given.ground, other.via, other.ground)
                if u == k or kind is None:
                    continue
                first.append(k)
                second.append(u)
                distances.append(
                    _spread(given, other.via, other.clearance)
                    if kind is PairKind.SPREAD
                    else apart[k][u]
                )
                kinds.append(kind)
        self.first = np.array(first, dtype=np.intp)
        self.second = np.array(second, dtype=np.intp)
        self.apart = np.array(distances, dtype=float)
        self.kinds = np.array(kinds, dtype=np.int8)
        self.weights = np.array([vias[k].via.diameter for k in first], dtype=float)
        self.slot = np.full((len(vias), len(vias)), _NO_PAIR, dtype=np.intp)
        """By the indices of a pair's first and second via, its index among
        the pairs."""
        self.slot[self.first, self.second] = np.arange(len(first))
        reach = obstacle_reach(params)
        cutoff = params.attraction_range if params.attraction else 0.0
        self.reach = max(
            (
                max(cutoff, a + reach) if kind is PairKind.RETURN else a + reach
                for a, kind in zip(distances, kinds, strict=True)
            ),
            default=0.0,
        )
        """The centre distance beyond which no pair of the problem's vias
        gives energy."""
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
                for k in range(len(vias))
            ],
            dtype=np.int8,
        ).reshape(len(vias), len(static))
        met = (static_kinds != _NO_PAIR).any(axis=0)
        self.static = [s for s, m in zip(static, met, strict=True) if m]
        self.static_kinds = static_kinds[:, met]
        self.static_apart = np.full(self.static_kinds.shape, math.nan)
        """By moving via and static via, the distance their centres keep,
        or for vias of one ground net the distance beyond which they repel;
        NaN until they first meet."""
        at = np.array([s.via.position for s in self.static], dtype=float)
        self.tree = cKDTree(at.reshape(-1, 2)) if self.static else None
        self.static_at = at.reshape(-1, 2)
        self.static_reach = np.array(
            [max(cutoff, self._widest(v) + reach) for v in vias], dtype=float
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
        near = self._near(at)
        first, second = self.first[near], self.second[near]
        apart, kinds, weights = self.apart[near], self.kinds[near], self.weights[near]
        value, push = self._terms(at[first] - at[second], apart, kinds, weights)
        _scatter(gradient, first, push)
        _scatter(gradient, second, -push)
        if self.tree is not None:
            rows, columns = self._static_near(at)
            static_value, static_push = self._terms(
                at[rows] - self.static_at[columns],
                self.static_apart[rows, columns],
                self.static_kinds[rows, columns],
                self.diameters[rows],
            )
            value += static_value
            _scatter(gradient, rows, static_push)
        return value, gradient

    def _near(self, at: FloatArray) -> IntArray:
        """The problem's pairs whose vias at `at` are near enough to give
        each other energy."""
        if not len(self.first):
            return np.zeros(0, dtype=np.intp)
        close = cKDTree(at).query_pairs(self.reach, output_type="ndarray")
        i, j = close[:, 0], close[:, 1]
        found = np.concatenate([self.slot[i, j], self.slot[j, i]])
        return np.sort(found[found != _NO_PAIR])

    def _static_near(self, at: FloatArray) -> tuple[IntArray, IntArray]:
        """The pairs of a moving via, by its row of `at`, and a static via
        it pairs with that is near enough to give it energy, with the
        distance they keep worked out."""
        assert self.tree is not None
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
            other = self.static[s]
            self.static_apart[n, s] = (
                _spread(self.vias[n], other.via, other.clearance)
                if self.static_kinds[n, s] == PairKind.SPREAD
                else self._kept(n, other)
            )
        return rows, columns

    def _kept(self, k: int, other: StaticVia) -> float:
        """The distance moving via k's centre keeps from `other`'s, as its
        static field would count it: with its zones on the layers `other`
        spans."""
        given = self.vias[k]
        clearances = {given.via.net: given.clearance, other.via.net: other.clearance}
        distance = via_via_inflation(
            given.via,
            other.via,
            clearances.__getitem__,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
            zones=given.zones,
        )
        # Paired vias share a layer, where vias of other nets keep their
        # copper clearance.
        assert distance is not None
        return distance

    def _widest(self, given: ViaInput) -> float:
        """The largest distance `given`'s centre keeps from a static via it
        pairs with, or beyond which one of its net repels it (`_spread`),
        which is no larger: from one as wide, as widely drilled and of as wide a
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
        spreading = kinds == PairKind.SPREAD
        value, slope = self.obstacle(r - apart)
        if spreading.any():
            value[spreading], slope[spreading] = self.repulsion(
                r[spreading] - apart[spreading]
            )
        attracting = kinds == PairKind.RETURN
        if attracting.any():
            pull, pull_slope = self.attraction(r[attracting], apart[attracting])
            value[attracting] += pull
            slope[attracting] += pull_slope
        units = np.divide(
            delta, r[:, None], out=np.zeros_like(delta), where=r[:, None] > 0.0
        )
        return float(weights @ value), (weights * slope)[:, None] * units
