import numpy as np
from distance_fields import BoolArray, FloatArray

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
