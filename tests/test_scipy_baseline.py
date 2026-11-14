"""The optimizer against the baseline
(`tests/data/scipy_baseline_darwin-arm64.json`, recorded on macOS arm64
as `scripts/record_baseline.py` says), which every platform compares
with, since the Rust optimizer computes the same bits on every platform.
The cases run at one spacing, as recorded (`record_baseline.SETTINGS`),
so they check that path: every case ends as recorded, converged where it
converged, its tracks no longer than recorded by more than the tolerance
below, and its copper clear: each chain and section of its static
copper, tracks of different nets of each other, and each moving via of
its static copper, the other moving vias and the chains.

Lengths are summed over a case, since a via that settles along a flat
valley trades length between the chains it joins. How near to clearance
copper ends is not compared with the record: any distance that keeps
clearance is as good.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import benchmark
import numpy as np
import pytest
from distance_fields import FAR
from record_baseline import PAD_ROWS, Record, pad_rows_record, scenario_run
from scenarios import ViasScenario
from shapely import LineString, Point

from kibends.bursts import DEFAULT_CHORD_TOLERANCE
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE
from kibends.optimize import RunResult
from kibends.via_moves import MovingVias

BASELINE = Path(__file__).parent / "data" / "scipy_baseline_darwin-arm64.json"
CASES = cast(dict[str, Record], json.loads(BASELINE.read_text())["cases"])

LENGTH_TOLERANCE = DEFAULT_CHORD_TOLERANCE
"""How much longer than recorded a case's tracks may end, mm: below the
chord tolerance, simplification may move the committed tracks as much."""

SLOW = frozenset({"via-grid-3x4", PAD_ROWS})
"""Cases that take a second or more."""


def entries(record: Record, key: str) -> Sequence[Record]:
    return cast(Sequence[Record], record.get(key, []))


def total_length(record: Record) -> float:
    chains = [cast(float, c["length"]) for c in entries(record, "chains")]
    pairs = [
        length
        for p in entries(record, "pairs")
        for length in cast(list[float], p["lengths"])
    ]
    return sum(chains) + sum(pairs)


def smallest_clearance(record: Record) -> float | None:
    """The smallest clearance to static copper of a case's chains and
    sections, None where all are in free space."""
    found = [
        cast(float | None, e["clearance"])
        for key in ("chains", "pairs")
        for e in entries(record, key)
    ]
    return min((c for c in found if c is not None), default=None)


def smallest_gap(record: Record) -> float | None:
    """The smallest copper gap between tracks of different nets, None
    where a case has none."""
    return cast(float | None, record.get("gap"))


type Position = tuple[float, float]


def via_clearances(
    scenario: ViasScenario, positions: Sequence[Position]
) -> list[float | None]:
    """Each moving via's smallest clearance at `positions`, as the
    optimizer measures it, to what a baseline record fixes: its static
    copper, the other moving vias and the chains that stand still; None
    where none of them is near."""
    moving = MovingVias(scenario.chains, scenario.vias, HOLE_CLEARANCE, HOLE_TO_HOLE)
    found: list[float | None] = []
    for v, (given, at) in enumerate(zip(scenario.vias, positions, strict=True)):
        static = float(given.copper.signed_distance(np.array([at]))[0][0])
        vias = [
            Point(at).distance(Point(positions[u])) - apart
            for u in range(len(positions))
            if u != v and (apart := moving.apart(v, u)) is not None
        ]
        still = [
            LineString(scenario.chains[c].centreline.points).distance(Point(at)) - ring
            for c, ring in moving.rings(v).items()
            if scenario.chains[c].fixed.all()
        ]
        near = [*([static] if static < FAR else []), *vias, *still]
        found.append(min(near, default=None))
    return found


def moving_track_clearances(scenario: ViasScenario, result: RunResult) -> list[float]:
    """Each moving via's smallest clearance in `result`, as the optimizer
    measures it, to the chains of other nets that move; none for a via
    near no such chain."""
    moving = MovingVias(scenario.chains, scenario.vias, HOLE_CLEARANCE, HOLE_TO_HOLE)
    found: list[float] = []
    for v, via in enumerate(result.vias):
        near = [
            LineString(result.chains[c].polyline).distance(Point(via.position)) - ring
            for c, ring in moving.rings(v).items()
            if not scenario.chains[c].fixed.all()
        ]
        found.extend([min(near)] if near else [])
    return found


def via_positions(record: Record) -> list[Position]:
    return [(x, y) for x, y in cast(list[list[float]], record["vias"])]


def clear(clearance: float | None) -> bool:
    """Whether a signed clearance margin keeps clearance; None is far from
    everything."""
    return clearance is None or clearance >= 0.0


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in CASES],
)
def test_a_case_ends_as_recorded(name: str) -> None:
    baseline = CASES[name]
    scenario = None if name == PAD_ROWS else benchmark.scenario_makers()[name]()

    if scenario is None:
        record = pad_rows_record()
    else:
        record, result = scenario_run(scenario)
        if isinstance(scenario, ViasScenario):
            assert isinstance(result, RunResult)
            assert all(map(clear, moving_track_clearances(scenario, result)))
            assert all(map(clear, via_clearances(scenario, via_positions(record))))

    assert record["termination"] == baseline["termination"]
    assert total_length(record) <= total_length(baseline) + LENGTH_TOLERANCE
    assert clear(smallest_clearance(record))
    assert clear(smallest_gap(record))
