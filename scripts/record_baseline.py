"""Record what the scipy optimizer gives, as the baseline that the Rust
optimizer's results are compared with.

Runs each scenario of `tests/scenarios.py` that `./run bench --only
scenarios` runs, at default settings, and the bench case "10 tracks between
pad rows at 1 mm" through a fake KiCad, and writes, for each, how the run
ended, its outer iterations and energy evaluations, each chain's length,
each coupled section's track lengths, each moving via's position, and the
clearances the scenario defines: each chain's and section's smallest signed
distance from its static field along its centreline, None in free space,
and the smallest copper gap between tracks, chains' and sections', of
different nets on one layer, other than the two nets of a differential
pair. The pad-row case records the lengths of the simplified tracks and no
clearances.

The file is `tests/data/scipy_baseline_<platform>.json`, the platform being
`sys.platform` and the machine, such as `darwin-arm64` or `linux-x86_64`,
since the optimizer's floating-point results differ between platforms.

Usage: ./run sync && .venv/bin/python scripts/record_baseline.py
(no arguments: the benchmark it imports reads the command line).
"""

import itertools
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

# benchmark puts tests/ and src/ on sys.path, so it is imported first.
import benchmark
from conftest import clearances
from distance_fields import FAR
from loguru import logger
from scenarios import PairsScenario, Scenario, ViasScenario
from shapely import LineString

from kibends.energy import DistanceField, FloatArray
from kibends.optimize import OptimizeResult

PAD_ROWS = "10 tracks between pad rows at 1 mm"

type Record = dict[str, object]


def platform_tag() -> str:
    return f"{sys.platform}-{platform.machine()}"


def clearance(polyline: FloatArray, field: DistanceField) -> float | None:
    """The smallest signed distance from `field` along `polyline`, None
    where the field holds no obstacle."""
    smallest = float(clearances(polyline, field).min())
    return None if smallest >= FAR else smallest


class Track(NamedTuple):
    layer: str
    net: str
    width: float
    points: FloatArray


def smallest_gap(
    tracks: list[Track], pair_of: dict[str, frozenset[str]]
) -> float | None:
    """The smallest copper gap between tracks of different nets on one
    layer, but between the two nets of a differential pair, whose nets
    `pair_of` gives, mm; None if there are no such tracks."""
    gaps = [
        LineString(a.points).distance(LineString(b.points)) - (a.width + b.width) / 2
        for a, b in itertools.combinations(tracks, 2)
        if a.layer == b.layer and a.net != b.net and b.net not in pair_of.get(a.net, ())
    ]
    return min(gaps, default=None)


def scenario_record(s: Scenario | benchmark.Several) -> Record:
    with benchmark.counted() as count:
        result = benchmark.run_scenario(s)
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
        return record
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
        Track(c.layer, c.centreline.net, c.centreline.width, p)
        for c, p in zip(s.chains, polylines, strict=True)
    ] + [
        Track(given.layer, track.net, track.width, points)
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
    return record


def pad_rows_record() -> Record:
    board = benchmark.pad_rows_board()
    options = replace(benchmark.smooth_options(), dry_run=True)
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


def timed(name: str, record: Callable[[], Record]) -> Record:
    began = time.perf_counter()
    result = record()
    print(f"{name}: {time.perf_counter() - began:.2f} s", flush=True)
    return result


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    cases: dict[str, Record] = {
        name: timed(name, lambda make=make: scenario_record(make()))
        for name, make in benchmark.scenario_makers().items()
    }
    cases[PAD_ROWS] = timed(PAD_ROWS, pad_rows_record)
    baseline = {
        "platform": platform_tag(),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "cases": cases,
    }
    path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "data"
        / f"scipy_baseline_{platform_tag()}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
