from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from conftest import Ticking
from distance_fields import FloatArray

import kibends.optimize as optimize_module
from kibends import _core
from kibends._core.testing import quadratic
from kibends.bursts import (
    DEFAULT_SETTINGS,
    FALLBACK_WARNING,
    OptimizerSettings,
    ScaledObjective,
    run_burst,
)

SHORT_SETTINGS = replace(DEFAULT_SETTINGS, burst_maxiter=10)

# How much steeper the valley's second coordinate is than its first.
STEEPNESS = 100.0

squared = quadratic([1.0, 1.0])

steep_valley = quadratic([1.0, STEEPNESS])
"""A quadratic that one L-BFGS iteration does not minimise."""


def nothing_crosses(x: FloatArray) -> int:
    return 0


def test_a_burst_without_coordinates_ends_where_it_is() -> None:
    burst = run_burst(squared, np.zeros(0), nothing_crosses, DEFAULT_SETTINGS)

    assert burst.points.shape == (0,)
    assert not burst.abnormal


def test_a_burst_reports_how_l_bfgs_ended_it() -> None:
    one_iteration = replace(DEFAULT_SETTINGS, burst_maxiter=1)
    start = np.array([1.0, 2.0])

    converged = run_burst(squared, start, nothing_crosses, DEFAULT_SETTINGS)
    capped = run_burst(steep_valley, start, nothing_crosses, one_iteration)

    assert converged.message.startswith("CONVERGENCE")
    assert capped.iterations == 1
    assert capped.message.startswith("STOP")


def test_a_gradient_below_gtol_ends_the_burst_before_its_first_iteration() -> None:
    settings = replace(DEFAULT_SETTINGS, lbfgs_gtol=10.0)

    burst = run_burst(squared, np.array([1.0, 2.0]), nothing_crosses, settings)

    assert burst.iterations == 0
    assert burst.message == "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"


def test_a_relative_decrease_below_ftol_ends_the_burst() -> None:
    # Every decrease of an energy that cannot fall below 0 is at most 1,
    # relatively.
    settings = replace(DEFAULT_SETTINGS, lbfgs_ftol=1.0)

    burst = run_burst(steep_valley, np.array([1.0, 2.0]), nothing_crosses, settings)

    assert burst.iterations == 1
    assert burst.message == "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"


def test_a_burst_whose_line_search_fails_ends_abnormally_where_it_started() -> None:
    start = np.array([1.0, 2.0])
    reverted: list[FloatArray] = []

    def checked(x: FloatArray) -> int:
        reverted.append(x.copy())
        return 0

    burst = run_burst(
        quadratic([1.0, 1.0], uphill=True), start, checked, SHORT_SETTINGS
    )

    assert burst.abnormal
    assert burst.message.startswith("ABNORMAL")
    np.testing.assert_array_equal(burst.points, start)
    np.testing.assert_array_equal(reverted[0], start)


def test_an_exception_from_stop_ends_the_burst_and_propagates() -> None:
    def interrupted() -> bool:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_burst(
            steep_valley,
            np.array([3.0, 2.0]),
            nothing_crosses,
            SHORT_SETTINGS,
            interrupted,
        )


VALLEY_SCALE = np.array([1.0, 1.0 / np.sqrt(STEEPNESS)])
"""The steep valley's coordinates scaled by 1/sqrt of its curvatures, so
that it is round."""
NEAR_MINIMUM = 1e-6
ABOVE_ANY_GRADIENT = 1e6
"""A gtol above the valley's gradient anywhere near its minimum, scaled or
not, so that a burst ends before its first step."""


def test_a_scaled_burst_checks_and_returns_the_coordinates_unscaled() -> None:
    # At the start, where the scaled coordinates differ from the unscaled
    # ones; at the minimum both are zero.
    start = np.array([1.0, 2.0])
    no_step = replace(DEFAULT_SETTINGS, lbfgs_gtol=ABOVE_ANY_GRADIENT)
    checked: list[FloatArray] = []

    def check(x: FloatArray) -> int:
        checked.append(x.copy())
        return 0

    burst = run_burst(
        ScaledObjective(steep_valley, VALLEY_SCALE), start, check, no_step
    )

    np.testing.assert_allclose(checked, [start])
    np.testing.assert_allclose(burst.points, start)


def test_a_burst_keeps_what_its_check_moves_back() -> None:
    start = np.array([1.0, 2.0])

    def second_back(x: FloatArray) -> int:
        x[1] = start[1]
        return 1

    scaled = ScaledObjective(steep_valley, VALLEY_SCALE)

    burst = run_burst(scaled, start, second_back, DEFAULT_SETTINGS)

    assert burst.points[1] == start[1]
    assert abs(burst.points[0]) < NEAR_MINIMUM


def test_a_scaled_objective_answers_as_its_objective() -> None:
    x = np.array([1.0, 2.0])
    scaled = ScaledObjective(steep_valley, VALLEY_SCALE)

    value, gradient = scaled(x)

    expected_value, expected_gradient = steep_valley(x)
    assert value == expected_value
    np.testing.assert_array_equal(gradient, expected_gradient)


@pytest.mark.fallback
def test_a_burst_moved_back_whole_is_logged_as_a_warning(log: list[str]) -> None:
    report = optimize_module._Report([], Ticking(), 0.0, 0.0)  # pyright: ignore[reportPrivateUsage]

    report(_core.FallbackNotice())

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
    whole = run_burst(steep_valley, start, nothing_crosses, SHORT_SETTINGS)

    burst = run_burst(
        steep_valley, start, nothing_crosses, SHORT_SETTINGS, ask_until(2)
    )

    assert whole.iterations > 2
    assert burst.iterations == 2
    assert burst.cut
    assert not burst.abnormal


def test_a_burst_its_stop_never_ends_runs_whole() -> None:
    start = np.array([3.0, 2.0])
    whole = run_burst(steep_valley, start, nothing_crosses, SHORT_SETTINGS)

    burst = run_burst(
        steep_valley, start, nothing_crosses, SHORT_SETTINGS, lambda: False
    )

    np.testing.assert_array_equal(burst.points, whole.points)
    assert burst.iterations == whole.iterations
    assert not burst.cut


def test_settings_take_coarse_spacings_above_the_final_one_coarsest_first() -> None:
    settings = OptimizerSettings(spacing=0.1, coarse_spacings=(1.6, 0.4))
    assert settings.coarse_spacings == (1.6, 0.4)
    assert settings.spacing == 0.1


@pytest.mark.parametrize(
    "coarse",
    [(0.1,), (0.05,), (0.4, 0.1), (0.4, 0.4), (0.4, 1.6)],
    ids=["equal", "below", "last-equal", "repeated", "increasing"],
)
def test_settings_refuse_coarse_spacings_not_falling_above_the_final_one(
    coarse: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="coarse_spacings"):
        OptimizerSettings(spacing=0.1, coarse_spacings=coarse)


def test_settings_refuse_more_coarse_spacings_than_the_optimizer_holds() -> None:
    most = tuple(0.2 * 2.0**k for k in reversed(range(_core.MAX_COARSE_LEVELS)))
    assert OptimizerSettings(coarse_spacings=most).coarse_spacings == most
    with pytest.raises(ValueError, match="coarse_spacings"):
        OptimizerSettings(coarse_spacings=(2.0 * most[0], *most))
