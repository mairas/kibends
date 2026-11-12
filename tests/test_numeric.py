"""`kibends._core.interp` against numpy's `interp` as the macOS arm64 wheel
computes it, its product fused into the sum."""

import bisect
import math
import platform
import sys

import numpy as np
import pytest

from kibends._core import interp
from kibends.energy import FloatArray

FUSED_NUMPY = sys.platform == "darwin" and platform.machine() == "arm64"
CASES = 300


def _reference(x: FloatArray, knots: FloatArray, values: FloatArray) -> FloatArray:
    xp: list[float] = knots.tolist()
    fp: list[float] = values.tolist()
    out: list[float] = []
    for value in x.tolist():
        j = bisect.bisect_right(xp, value) - 1
        if math.isnan(value):
            out.append(value)
        elif j < 0:
            out.append(fp[0])
        elif j == len(xp) - 1 or xp[j] == value:
            out.append(fp[j])
        else:
            slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
            out.append(math.fma(slope, value - xp[j], fp[j]))
    return np.array(out)


def _cases() -> list[tuple[FloatArray, FloatArray, FloatArray]]:
    rng = np.random.default_rng(1)
    cases: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for _ in range(CASES):
        knots = int(rng.integers(2, 30))
        xp = np.sort(rng.random(knots) * 10.0)
        fp = rng.normal(size=knots) * 5.0
        x = np.concatenate([rng.random(int(rng.integers(1, 60))) * 12.0 - 1.0, xp])
        cases.append((x, xp, fp))
    return cases


def test_interp_fuses_like_the_reference() -> None:
    for x, xp, fp in _cases():
        assert np.array_equal(interp(x, xp, fp), _reference(x, xp, fp))


@pytest.mark.skipif(not FUSED_NUMPY, reason="numpy fuses interp on macOS arm64 only")
def test_interp_is_numpy_s_on_macos_arm64() -> None:
    for x, xp, fp in _cases():
        assert np.array_equal(interp(x, xp, fp), np.interp(x, xp, fp))


def test_interp_passes_nan_through() -> None:
    result = interp(np.array([np.nan, 0.5]), np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert np.isnan(result[0])
    assert result[1] == 1.0


def test_interp_refuses_knots_without_values() -> None:
    with pytest.raises(ValueError, match="unequal"):
        interp(np.array([0.5]), np.array([0.0, 1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="no knot"):
        interp(np.array([0.5]), np.array([]), np.array([]))
