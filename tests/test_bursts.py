from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import scenarios
from distance_fields import BoolArray, FloatArray
from scipy.optimize import OptimizeResult

import kibends.bursts as bursts_module
from kibends.bursts import (
    DEFAULT_SETTINGS,
    FALLBACK_WARNING,
    revert_until_clear,
    run_burst,
)
from kibends.energy import total_energy
from kibends.polylines import resample

# A burst of the wrap at these settings takes a few milliseconds.
WRAP_SETTINGS = replace(DEFAULT_SETTINGS, spacing=0.5, burst_maxiter=10)

# How much steeper the valley's second coordinate is than its first.
STEEPNESS = 100.0


def squared(x: FloatArray) -> tuple[float, FloatArray]:
    return float(x @ x), 2 * x


def steep_valley(x: FloatArray) -> tuple[float, FloatArray]:
    """A quadratic that one L-BFGS-B iteration does not minimise."""
    scale = np.array([1.0, STEEPNESS])
    return float(scale @ x**2), 2 * scale * x


def nothing_crosses(x: FloatArray) -> int:
    return 0


def test_a_burst_without_coordinates_ends_where_it_is() -> None:
    burst = run_burst(squared, np.zeros(0), nothing_crosses, DEFAULT_SETTINGS)

    assert burst.points.shape == (0,)
    assert not burst.abnormal


def test_a_burst_without_coordinates_reports_no_work() -> None:
    burst = run_burst(squared, np.zeros(0), nothing_crosses, DEFAULT_SETTINGS)

    assert (burst.iterations, burst.evaluations) == (0, 0)


def test_a_burst_reports_its_iterations_and_evaluations() -> None:
    scenario = scenarios.wrap()
    points, fixed = resample(scenario.polyline, scenario.fixed, WRAP_SETTINGS.spacing)
    calls = 0

    def energy(x: FloatArray) -> tuple[float, FloatArray]:
        nonlocal calls
        calls += 1
        moved = points.copy()
        moved[~fixed] = x.reshape(-1, 2)
        value, gradient = total_energy(moved, fixed, scenario.field, scenario.params)
        return value, gradient[~fixed].ravel()

    burst = run_burst(energy, points[~fixed].ravel(), nothing_crosses, WRAP_SETTINGS)

    assert 1 <= burst.iterations <= WRAP_SETTINGS.burst_maxiter
    assert burst.evaluations >= burst.iterations
    assert burst.evaluations == calls


def test_a_burst_reports_how_l_bfgs_b_ended_it() -> None:
    one_iteration = replace(DEFAULT_SETTINGS, burst_maxiter=1)
    start = np.array([1.0, 2.0])

    converged = run_burst(squared, start, nothing_crosses, DEFAULT_SETTINGS)
    capped = run_burst(steep_valley, start, nothing_crosses, one_iteration)

    assert converged.message.startswith("CONVERGENCE")
    assert capped.iterations == 1
    assert capped.message.startswith("STOP")


def test_lbfgs_tolerances_reach_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    real = cast(Callable[..., OptimizeResult], bursts_module.minimize)

    def spy(*args: object, **kwargs: object) -> OptimizeResult:
        seen.update(cast(dict[str, object], kwargs["options"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(bursts_module, "minimize", spy)
    settings = replace(DEFAULT_SETTINGS, lbfgs_ftol=1e-8, lbfgs_gtol=1e-6)

    run_burst(squared, np.array([1.0, 2.0]), nothing_crosses, settings)

    assert (seen["ftol"], seen["gtol"]) == (1e-8, 1e-6)


@pytest.mark.fallback
def test_crossings_naming_nothing_to_move_back_move_all_back_and_warn(
    log: list[str],
) -> None:
    start = np.zeros(4)
    candidate = np.ones(4)
    stuck = np.zeros(4, dtype=bool)

    def crossing_until_back(x: FloatArray) -> tuple[int, BoolArray]:
        return int(x.any()), stuck

    reverted, count = revert_until_clear(candidate, start, crossing_until_back)

    assert count == 1
    assert reverted.all()
    assert np.array_equal(candidate, start)
    assert [line.strip() for line in log] == [f"WARNING {FALLBACK_WARNING}"]
