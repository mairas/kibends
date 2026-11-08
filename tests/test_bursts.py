from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from distance_fields import BoolArray, FloatArray
from scipy.optimize import OptimizeResult

import kibends.bursts as bursts_module
from kibends.bursts import DEFAULT_SETTINGS, bounded_burst

ROUNDED_AWAY = 1e-17
"""A bound below the float spacing at 1 mm, so that it equals the start."""


def squared(x: FloatArray) -> tuple[float, FloatArray]:
    return float(x @ x), 2 * x


def nothing_crosses(x: FloatArray) -> tuple[BoolArray, int]:
    return np.zeros(len(x), dtype=bool), 0


def test_bounds_equal_to_the_start_in_floating_point_leave_it_unchanged() -> None:
    start = np.array([1.0, 2.0])

    burst = bounded_burst(
        squared,
        start,
        np.full(len(start), ROUNDED_AWAY),
        nothing_crosses,
        DEFAULT_SETTINGS,
    )

    np.testing.assert_array_equal(burst.points, start)
    assert not burst.abnormal
    assert burst.rejected_edges == 0


def test_lbfgs_tolerances_reach_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    real = cast(Callable[..., OptimizeResult], bursts_module.minimize)

    def spy(*args: object, **kwargs: object) -> OptimizeResult:
        seen.update(cast(dict[str, object], kwargs["options"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(bursts_module, "minimize", spy)
    settings = replace(DEFAULT_SETTINGS, lbfgs_ftol=1e-9, lbfgs_gtol=1e-6)

    bounded_burst(squared, np.array([1.0, 2.0]), np.ones(2), nothing_crosses, settings)

    assert (seen["ftol"], seen["gtol"]) == (1e-9, 1e-6)
