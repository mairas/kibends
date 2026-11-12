"""The optimizer against the scipy baseline
(`tests/data/scipy_baseline_darwin-arm64.json`, recorded on macOS arm64
while scipy ran the optimizer, as `scripts/record_baseline.py` says),
which every platform compares with, since the Rust optimizer computes the
same on every platform but for the last bits of libm: every case ends as
it did under scipy, converged where it converged, its tracks no longer and
no nearer to copper than scipy's by more than the tolerances below.

Lengths are summed over a case, since a via that settles along a flat
valley trades length between the chains it joins: per chain, the lengths
differ from scipy's by up to 0.1 mm on darwin-arm64, in sum by at most
9.4 µm (pair-turning-back-150) and elsewhere by 1 µm. Shorter or farther
from copper than scipy is no failure: on via-hemmed scipy itself ended
0.51 mm shorter on linux-x86_64 than on darwin-arm64.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import benchmark
import pytest
from record_baseline import PAD_ROWS, Record, pad_rows_record
from record_baseline import scenario_record as record_of

from kibends.bursts import DEFAULT_CHORD_TOLERANCE, DEFAULT_SETTINGS

BASELINE = Path(__file__).parent / "data" / "scipy_baseline_darwin-arm64.json"
CASES = cast(dict[str, Record], json.loads(BASELINE.read_text())["cases"])

LENGTH_TOLERANCE = DEFAULT_CHORD_TOLERANCE
"""How much longer than scipy's a case's tracks may end, mm: below the
chord tolerance, simplification may move the committed tracks as much."""
CLEARANCE_TOLERANCE = DEFAULT_SETTINGS.settled_change
"""How much nearer to copper than scipy's a case's tracks may end, mm: a
move a problem settles below. Measured on darwin-arm64: at most 1.7 µm
nearer (crowded's gap)."""

SLOW = frozenset({"via-grid-3x4", PAD_ROWS})
"""Cases that take a second or more."""


def run_case(name: str) -> Record:
    if name == PAD_ROWS:
        return pad_rows_record()
    return record_of(benchmark.scenario_makers()[name]())


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


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in CASES],
)
def test_a_case_ends_as_under_scipy(name: str) -> None:
    baseline = CASES[name]

    record = run_case(name)

    assert record["termination"] == baseline["termination"]
    assert total_length(record) <= total_length(baseline) + LENGTH_TOLERANCE
    for measure in (smallest_clearance, smallest_gap):
        then, now = measure(baseline), measure(record)
        assert (then is None) == (now is None)
        if then is not None and now is not None:
            assert now >= then - CLEARANCE_TOLERANCE
