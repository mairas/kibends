"""Temporary: the Rust run loop against the Python one it replaces
(`kibends._optimize_py`), bitwise, on the benchmark's scenarios, with a
time limit and with holds. Deleted with the Python loop."""

import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType

import benchmark
import numpy as np
import pytest
from conftest import Ticking
from loguru import logger
from scenarios import PairsScenario, Scenario, ViasScenario

from kibends import _core, _optimize_py
from kibends import optimize as rust_module
from kibends.bursts import DEFAULT_SETTINGS, OptimizerSettings
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline

from kibends.results import PairResult, RunResult

TIMING = re.compile(r"[0-9.]+ s in evaluations, [0-9.]+ s in the rest")


@contextmanager
def heard() -> Iterator[list[str]]:
    lines: list[str] = []

    def sink(message: object) -> None:
        lines.append(TIMING.sub("…", str(message).rstrip("\n")))

    handle = logger.add(sink, level="DEBUG", format="{level} {message}")
    try:
        yield lines
    finally:
        logger.remove(handle)


def inputs(s: Scenario | benchmark.Several) -> dict[str, object]:
    if isinstance(s, Scenario):
        alone = ChainInput(
            ChainCentreline("chain", "", 0.0, 0.0, s.polyline), "", s.fixed, s.field
        )
        return {"chains": [alone], "params": s.params}
    return {
        "chains": s.chains,
        "params": s.params,
        "vias": s.vias if isinstance(s, ViasScenario) else (),
        "pairs": s.pairs if isinstance(s, PairsScenario) else (),
    }


def run(
    module: ModuleType,
    s: Scenario | benchmark.Several,
    settings: OptimizerSettings,
    to_hold: Callable[[Sequence[PairResult]], Sequence[int]] | None,
    clock_step: float,
) -> tuple[RunResult, int, list[str]]:
    _core.reset_statistics()
    with heard() as lines:
        result = module.optimize_chains(
            **inputs(s),
            settings=settings,
            clock=Ticking(step=clock_step),
            to_hold=to_hold,
        )
    return result, _core.statistics().evaluations, lines


def assert_same(a: RunResult, b: RunResult) -> None:
    assert a.termination is b.termination
    assert a.iterations == b.iterations
    assert a.gave_up == b.gave_up
    assert a.abnormal == b.abnormal
    for x, y in zip(a.chains, b.chains, strict=True):
        np.testing.assert_array_equal(x.polyline, y.polyline)
        np.testing.assert_array_equal(x.fixed, y.fixed)
        np.testing.assert_array_equal(x.contact_of, y.contact_of)
        assert x.rejected_edges == y.rejected_edges
        assert x.before == y.before
    for x, y in zip(a.vias, b.vias, strict=True):
        assert x == y
    for x, y in zip(a.pairs, b.pairs, strict=True):
        np.testing.assert_array_equal(x.centreline, y.centreline)
        np.testing.assert_array_equal(x.fixed, y.fixed)
        np.testing.assert_array_equal(x.a, y.a)
        np.testing.assert_array_equal(x.b, y.b)
        assert x.rejected_edges == y.rejected_edges
        assert x.bend == y.bend
        assert x.block.hold == y.block.hold


def hold_first(count: list[int]) -> Callable[[Sequence[PairResult]], Sequence[int]]:
    def to_hold(results: Sequence[PairResult]) -> Sequence[int]:
        count[0] += 1
        return [0] if results and count[0] == 1 else []

    return to_hold


CASES = {
    **{name: (name, DEFAULT_SETTINGS, False, 0.0) for name in benchmark.scenario_makers()},
    **{
        f"{name}-timed": (name, replace(DEFAULT_SETTINGS, time_limit=3.5), False, 1.0)
        for name in ("tent", "via-detour", "pair-corner")
    },
    **{
        f"{name}-held": (name, DEFAULT_SETTINGS, True, 0.0)
        for name in ("pair-corner", "pair-tent", "pair-split", "pair-round-corner")
    },
}


@pytest.mark.slow
@pytest.mark.fallback
@pytest.mark.parametrize("case", CASES)
def test_the_rust_loop_is_the_python_loop(case: str) -> None:
    name, settings, holding, step = CASES[case]
    make = benchmark.scenario_makers()[name]
    python_holds: list[int] = [0]
    rust_holds: list[int] = [0]
    expected, expected_evaluations, expected_log = run(
        _optimize_py, make(), settings, hold_first(python_holds) if holding else None, step
    )
    got, evaluations, log = run(
        rust_module, make(), settings, hold_first(rust_holds) if holding else None, step
    )
    assert_same(got, expected)
    assert evaluations == expected_evaluations
    assert python_holds == rust_holds
    assert log == expected_log


BUSES = {
    "z-equal": lambda: benchmark.z_bus(0.0),
    "short-chains": benchmark.short_chains,
    "one-net": benchmark.one_net_chains,
}


@pytest.mark.slow
@pytest.mark.parametrize("bus", BUSES)
def test_the_rust_loop_is_the_python_loop_on_buses(bus: str) -> None:
    settings = replace(benchmark.smooth_options().settings, time_limit=None)
    if bus != "z-equal":
        settings = replace(settings, max_iterations=1)
    params = replace(benchmark.scenarios.PARAMS, repulsion=benchmark.BUS_REPULSION)
    outcomes: list[tuple[RunResult, int, list[str]]] = []
    for module in (_optimize_py, rust_module):
        _core.reset_statistics()
        with heard() as lines:
            result = module.optimize_chains(
                BUSES[bus](), params, settings, clock=Ticking()
            )
        outcomes.append((result, _core.statistics().evaluations, lines))
    (expected, expected_evaluations, expected_log), (got, evaluations, log) = outcomes
    assert_same(got, expected)
    assert evaluations == expected_evaluations
    assert log == expected_log


@pytest.mark.slow
def test_the_rust_loop_is_the_python_loop_between_pad_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[RunResult] = []
    for module in (_optimize_py, rust_module):

        def kept(*args: object, **kwargs: object) -> RunResult:
            result = module.optimize_chains(*args, **kwargs)
            outcomes.append(result)
            return result

        monkeypatch.setattr(benchmark.pipeline, "optimize_chains", kept)
        options = replace(benchmark.smooth_options(), dry_run=True)
        benchmark.smooth_board(benchmark.pad_rows_board(), options)
    expected, got = outcomes
    assert_same(got, expected)
