import numpy as np
from distance_fields import BoolArray, Disc, FloatArray, FreeSpace
from scenarios import PARAMS, chain, path

from kibends.optimize import OptimizerSettings, measure, optimize
from kibends.simplify import simplify

SPACING = 0.1
DISC = Disc(0.0, 0.0, 1.0)
WRAP_STANDOFF = 0.02
WIDE_TOLERANCE = 0.05


def free(n: int) -> BoolArray:
    fixed = np.zeros(n, dtype=bool)
    fixed[[0, -1]] = True
    return fixed


def arc_around_disc(standoff: float) -> FloatArray:
    radius = DISC.radius + standoff
    angles = np.linspace(0.0, np.pi, int(np.ceil(np.pi * radius / SPACING)) + 1)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def test_collinear_subsegments_collapse_to_one() -> None:
    polyline = path((0.0, 0.0), (1.0, 1.0), (2.5, 2.5), (3.0, 3.0))

    result = simplify(polyline, free(4), FreeSpace())

    assert np.array_equal(result, polyline[[0, -1]])


def test_never_merges_across_a_fixed_vertex() -> None:
    polyline = path((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    fixed = np.array([True, False, True, False, True])

    result = simplify(polyline, fixed, FreeSpace())

    assert np.array_equal(result, polyline[[0, 2, 4]])


def test_keeps_vertices_beyond_the_chord_tolerance() -> None:
    polyline = path((0.0, 0.0), (1.0, 0.1), (2.0, 0.0))

    result = simplify(polyline, free(3), FreeSpace(), tolerance=0.05)

    assert np.array_equal(result, polyline)


class TestWrapAroundDisc:
    def test_merges_only_where_the_chord_keeps_clearance(self) -> None:
        arc = arc_around_disc(WRAP_STANDOFF)

        tolerance_only = simplify(arc, free(len(arc)), FreeSpace(), WIDE_TOLERANCE)
        result = simplify(arc, free(len(arc)), DISC, WIDE_TOLERANCE)

        assert measure(tolerance_only, DISC, SPACING).min_clearance < 0.0
        assert len(tolerance_only) < len(result) < len(arc)
        assert DISC.segments_clear(result[:-1], result[1:]).all()
        assert measure(result, DISC, SPACING).min_clearance >= 0.0

    def test_unmergeable_wrap_is_left_alone(self) -> None:
        arc = arc_around_disc(standoff=0.001)

        result = simplify(arc, free(len(arc)), DISC, WIDE_TOLERANCE)

        assert np.array_equal(result, arc)


def test_optimized_straight_line_simplifies_to_one_segment() -> None:
    polyline, fixed = chain((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    # Leaves a bow of about 15 µm for simplify to flatten, as the defaults do.
    settings = OptimizerSettings(spacing=0.2, burst_maxiter=100)
    optimized = optimize(polyline, fixed, FreeSpace(), PARAMS, settings)

    result = simplify(optimized.polyline, optimized.fixed, FreeSpace())

    assert np.array_equal(result, polyline[[0, -1]])
