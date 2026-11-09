from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from distance_fields import FloatArray
from scipy.optimize import OptimizeResult

import kibends.bursts as bursts_module
from kibends.bursts import DEFAULT_SETTINGS, run_burst


def squared(x: FloatArray) -> tuple[float, FloatArray]:
    return float(x @ x), 2 * x


def nothing_crosses(x: FloatArray) -> int:
    return 0


def test_a_burst_without_coordinates_ends_where_it_is() -> None:
    burst = run_burst(squared, np.zeros(0), nothing_crosses, DEFAULT_SETTINGS)

    assert burst.points.shape == (0,)
    assert not burst.abnormal


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
