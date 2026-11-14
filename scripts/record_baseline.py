"""How the baseline is recorded, which `tests/test_scipy_baseline.py`
reuses to run its cases.

`main` records the cases on macOS arm64 into
`tests/data/scipy_baseline_darwin-arm64.json`, the one reference every
platform compares with: the Rust optimizer computes the same bits on every
platform. The cases are each scenario of `tests/scenarios.py` that `./run
bench --only scenarios` runs, and the bench case "10 tracks between pad
rows at 1 mm" through a fake KiCad, each at the defaults without coarse
levels (`SETTINGS`), a check of that path. For each, a record holds how the
run ended, its outer iterations and energy evaluations, each chain's
length, each coupled section's track lengths, each moving via's position,
and the clearances the scenario defines: each chain's and section's
smallest signed distance from its static field along its centreline, None
in free space, and the smallest copper gap between tracks, chains' and
sections', of different nets on one layer, other than the two nets of a
differential pair, less the larger of their clearances. The pad-row case
records the lengths of the simplified tracks and no clearances.
"""

import itertools
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

# benchmark puts tests/ and src/ on sys.path, so it is imported first.
import benchmark
import numpy
from conftest import clearances
from distance_fields import FAR
from loguru import logger
from scenarios import PairsScenario, Scenario, ViasScenario
from shapely import LineString

from kibends.bursts import DEFAULT_SETTINGS
from kibends.energy import DistanceField, FloatArray
from kibends.optimize import OptimizeResult, RunResult

PAD_ROWS = "10 tracks between pad rows at 1 mm"

SETTINGS = replace(DEFAULT_SETTINGS, coarse_spacings=())
"""The defaults at the final spacing alone, as the baseline was recorded."""

type Record = dict[str, object]


def clearance(polyline: FloatArray, field: DistanceField) -> float | None:
    """The smallest signed distance from `field` along `polyline`, None
    where the field holds no obstacle."""
    smallest = float(clearances(polyline, field).min())
    return None if smallest >= FAR else smallest


class Track(NamedTuple):
    layer: str
    net: str
    width: float
    clearance: float
    points: FloatArray


def smallest_gap(
    tracks: list[Track], pair_of: dict[str, frozenset[str]]
) -> float | None:
    """The smallest copper gap between tracks of different nets on one
    layer, but between the two nets of a differential pair, whose nets
    `pair_of` gives, less the larger of the two tracks' clearances, mm:
    negative inside clearance; None if there are no such tracks."""
    gaps = [
        LineString(a.points).distance(LineString(b.points))
        - (a.width + b.width) / 2
        - max(a.clearance, b.clearance)
        for a, b in itertools.combinations(tracks, 2)
        if a.layer == b.layer and a.net != b.net and b.net not in pair_of.get(a.net, ())
    ]
    return min(gaps, default=None)


def scenario_run(
    s: Scenario | benchmark.Several,
) -> tuple[Record, OptimizeResult | RunResult]:
    """The scenario's record and the result it was made from."""
    with benchmark.counted() as count:
        result = benchmark.run_scenario(s, SETTINGS)
    record: Record = {
        "termination": result.termination.value,
        "iterations": result.iterations,
        "evaluations": count.evaluations,
    }
    if isinstance(result, OptimizeResult):
        assert isinstance(s, Scenario)
        record["chains"] = [
            {
                "length": benchmark.length(result.polyline),
                "clearance": clearance(result.polyline, s.field),
            }
        ]
        return record, result
    assert not isinstance(s, Scenario)
    polylines = [c.polyline for c in result.chains]
    record["chains"] = [
        {"length": benchmark.length(p), "clearance": clearance(p, c.field)}
        for c, p in zip(s.chains, polylines, strict=True)
    ]
    pairs = s.pairs if isinstance(s, PairsScenario) else ()
    pair_of = {
        net: frozenset((given.a.net, given.b.net))
        for given in pairs
        for net in (given.a.net, given.b.net)
    }
    tracks = [
        Track(c.layer, c.centreline.net, c.centreline.width, c.centreline.clearance, p)
        for c, p in zip(s.chains, polylines, strict=True)
    ] + [
        Track(given.layer, track.net, track.width, track.clearance, points)
        for p, given in zip(result.pairs, pairs, strict=True)
        for track, points in zip((given.a, given.b), p.kept(), strict=True)
    ]
    record["gap"] = smallest_gap(tracks, pair_of)
    if pairs:
        record["pairs"] = [
            {
                "lengths": [benchmark.length(t) for t in p.kept()],
                "clearance": clearance(p.centreline, given.field),
            }
            for p, given in zip(result.pairs, pairs, strict=True)
        ]
    if isinstance(s, ViasScenario):
        record["vias"] = [list(v.position) for v in result.vias]
    return record, result


def pad_rows_record() -> Record:
    board = benchmark.pad_rows_board()
    options = benchmark.smooth_options()
    options = replace(
        options,
        dry_run=True,
        settings=replace(options.settings, coarse_spacings=SETTINGS.coarse_spacings),
    )
    with benchmark.counted() as count:
        report = benchmark.smooth_board(board, options)
    return {
        "termination": report.termination.value,
        "iterations": report.iterations,
        "evaluations": count.evaluations,
        "chains": [
            {"length": c.after.length, "clearance": None} for c in report.chains
        ],
    }


DATA = Path(__file__).parents[1] / "tests" / "data"
BASELINE = DATA / "scipy_baseline_darwin-arm64.json"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    cases = {
        name: scenario_run(make())[0]
        for name, make in benchmark.scenario_makers().items()
    }
    cases[PAD_ROWS] = pad_rows_record()
    baseline = {
        "platform": "darwin-arm64",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "cases": cases,
    }
    BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
    print(BASELINE)


if __name__ == "__main__":
    main()
