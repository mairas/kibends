from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import scenarios
from conftest import total_energy
from distance_fields import BoolArray, FloatArray
from scipy.optimize import OptimizeResult

import kibends.bursts as bursts_module
from kibends.bursts import (
    DEFAULT_SETTINGS,
    FALLBACK_WARNING,
    ScaledObjective,
    revert_until_clear,
    run_burst,
)
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


VALLEY_SCALE = np.array([1.0, 1.0 / np.sqrt(STEEPNESS)])
"""The steep valley's coordinates scaled by 1/sqrt of its curvatures, so
that it is round."""
NEAR_MINIMUM = 1e-6


def test_a_scale_that_rounds_the_valley_reaches_its_minimum_in_two_iterations() -> None:
    # The first step is a line search along the gradient, which in a round
    # valley points at the minimum; the second lands on it.
    two_iterations = replace(DEFAULT_SETTINGS, burst_maxiter=2)
    start = np.array([1.0, 2.0])
    scaled = ScaledObjective(steep_valley, VALLEY_SCALE)

    plain = run_burst(steep_valley, start, nothing_crosses, two_iterations)
    burst = run_burst(scaled, start, nothing_crosses, two_iterations)

    assert np.abs(plain.points).max() > NEAR_MINIMUM
    assert np.abs(burst.points).max() < NEAR_MINIMUM


def test_a_scaled_burst_checks_and_returns_the_coordinates_unscaled() -> None:
    start = np.array([1.0, 2.0])
    checked: list[FloatArray] = []

    def first_back(x: FloatArray) -> int:
        checked.append(x.copy())
        x[0] = start[0]
        return 1

    burst = run_burst(
        ScaledObjective(steep_valley, VALLEY_SCALE), start, first_back, DEFAULT_SETTINGS
    )

    assert np.abs(checked[0]).max() < NEAR_MINIMUM
    assert burst.points[0] == start[0]
    assert abs(burst.points[1]) < NEAR_MINIMUM


def test_a_scaled_objective_answers_as_its_objective() -> None:
    x = np.array([1.0, 2.0])
    scaled = ScaledObjective(steep_valley, VALLEY_SCALE)

    value, gradient = scaled(x)

    expected_value, expected_gradient = steep_valley(x)
    assert value == expected_value
    np.testing.assert_array_equal(gradient, expected_gradient)


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


def ask_until(answers: int) -> Callable[[], bool]:
    """A stop that answers True from its `answers`th question on."""
    asked = 0

    def stop() -> bool:
        nonlocal asked
        asked += 1
        return asked >= answers

    return stop


def test_stop_ends_the_burst_at_the_iteration_it_answers_true() -> None:
    start = np.array([3.0, 2.0])
    whole = run_burst(steep_valley, start, nothing_crosses, WRAP_SETTINGS)

    burst = run_burst(steep_valley, start, nothing_crosses, WRAP_SETTINGS, ask_until(2))

    assert whole.iterations > 2
    assert burst.iterations == 2
    assert burst.cut
    assert not burst.abnormal


def test_a_burst_its_stop_never_ends_runs_whole() -> None:
    start = np.array([3.0, 2.0])
    whole = run_burst(steep_valley, start, nothing_crosses, WRAP_SETTINGS)

    burst = run_burst(
        steep_valley, start, nothing_crosses, WRAP_SETTINGS, lambda: False
    )

    np.testing.assert_array_equal(burst.points, whole.points)
    assert burst.iterations == whole.iterations
    assert not burst.cut
