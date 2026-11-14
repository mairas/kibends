"""The analytic segment checks agree with dense sampling of the distance."""

import numpy as np
import pytest
from distance_fields import Box, Disc, FloatArray, Union, Walls

SEGMENT_COUNT = 400
SAMPLES_PER_SEGMENT = 2001
GRAZE = 1e-4
"""mm: a segment that grazes a shape by less than this can pass between
the samples, so sampling cannot judge it."""

SHAPES = [
    Disc(0.0, 0.0, 1.0),
    Box(0.0, 0.0, 1.0, 0.6),
    Box(0.0, 0.0, 1.0, 0.6, corner_radius=0.3),
    Walls(half_gap=1.0),
    Union([Disc(-1.0, 0.0, 0.5), Box(1.0, 0.0, 0.4, 0.8)]),
]


def sampled_minimum(
    shape: Disc | Box | Walls | Union, starts: FloatArray, ends: FloatArray
) -> FloatArray:
    t = np.linspace(0.0, 1.0, SAMPLES_PER_SEGMENT)
    points = starts[:, None, :] + t[None, :, None] * (ends - starts)[:, None, :]
    distance, _ = shape.signed_distance(points.reshape(-1, 2))
    return np.asarray(distance.reshape(len(starts), -1).min(axis=1))


@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: type(shape).__name__)
def test_segments_clear_matches_sampling(shape: Disc | Box | Walls | Union) -> None:
    rng = np.random.default_rng(7)
    starts = rng.uniform(-2.5, 2.5, (SEGMENT_COUNT, 2))
    ends = starts + rng.uniform(-1.5, 1.5, (SEGMENT_COUNT, 2))

    clear = shape.segments_clear(starts, ends)
    minimum = sampled_minimum(shape, starts, ends)

    decided = np.abs(minimum) > GRAZE
    assert np.array_equal(clear[decided], minimum[decided] > 0.0)
    assert clear.any() and not clear.all()


def test_segment_along_a_sharp_box_side_is_clear() -> None:
    box = Box(0.0, 0.0, 1.0, 1.0)
    starts = np.array([[-2.0, 1.0], [-2.0, 0.0]])
    ends = np.array([[2.0, 1.0], [2.0, 0.0]])
    assert box.segments_clear(starts, ends).tolist() == [True, False]
