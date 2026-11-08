import math

import numpy as np
import pytest
from distance_fields import FloatArray, FreeSpace

from kibends.neighbours import ChainInput, neighbour_centrelines
from kibends.obstacles import ChainCentreline

WIDTH = 0.2
REACH = 0.1
CLEAR = WIDTH + REACH
LEG = 5.0
ANGLE = math.radians(30.0)


def chain(name: str, *xy: tuple[float, float]) -> ChainInput:
    points = np.array(xy)
    fixed = np.zeros(len(points), dtype=bool)
    fixed[[0, -1]] = True
    return ChainInput(
        ChainCentreline(name, "N", WIDTH, 0.2, points), "F.Cu", fixed, FreeSpace()
    )


def kept_start(own: ChainInput, other: ChainInput) -> FloatArray:
    (part,) = neighbour_centrelines(own, other, other.centreline.points, REACH)
    return part.points[0]


OWN = chain("own", (0.0, 0.0), (LEG, 0.0))


@pytest.mark.parametrize("offset", [0.0, 0.1, 0.19])
def test_a_neighbour_leaving_a_touch_is_kept_from_where_it_clears(
    offset: float,
) -> None:
    # Pins `offset` apart overlap at WIDTH; beyond the left-out part the
    # neighbour lies at least CLEAR from this chain, and no farther at its start.
    other = chain(
        "other",
        (0.0, offset),
        (LEG * math.cos(ANGLE), offset + LEG * math.sin(ANGLE)),
    )

    start = kept_start(OWN, other)

    assert start[1] == pytest.approx(CLEAR)


def test_a_stub_ending_beside_a_chain_is_kept_from_where_it_clears() -> None:
    bar = chain("bar", (0.0, 0.0), (2.0, 0.0), (4.0, 0.0))
    bar.fixed[1] = True
    stub = chain("stub", (2.0, -3.0), (2.0, -0.1))

    (part,) = neighbour_centrelines(bar, stub, stub.centreline.points, REACH)
    left, right = neighbour_centrelines(stub, bar, bar.centreline.points, REACH)

    assert part.points[-1] == pytest.approx((2.0, -CLEAR))
    assert math.dist(left.points[-1], (2.0, -0.1)) == pytest.approx(CLEAR)
    assert math.dist(right.points[0], (2.0, -0.1)) == pytest.approx(CLEAR)


def test_a_neighbour_departing_alongside_is_left_out_on_that_side() -> None:
    own = chain("own", (0.0, 0.0), (2 * LEG, 0.0))
    alongside = chain("alongside", (0.0, 0.1), (LEG, 0.1))

    parts = neighbour_centrelines(own, alongside, alongside.centreline.points, REACH)

    assert parts == []


def test_a_neighbour_touched_twice_is_kept_around_both_touches() -> None:
    bar = chain("bar", (0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0))
    bar.fixed[[1, 2]] = True
    tips = ((2.0, -0.1), (4.0, -0.1))
    arch = chain("arch", tips[0], (2.0, -3.0), (4.0, -3.0), tips[1])

    parts = neighbour_centrelines(arch, bar, bar.centreline.points, REACH)

    assert len(parts) == 3
    first, middle, last = (p.points for p in parts)
    assert math.dist(first[-1], tips[0]) == pytest.approx(CLEAR)
    assert math.dist(middle[0], tips[0]) == pytest.approx(CLEAR)
    assert math.dist(middle[-1], tips[1]) == pytest.approx(CLEAR)
    assert math.dist(last[0], tips[1]) == pytest.approx(CLEAR)
