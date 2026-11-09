"""Vertices that keep a pad or via (`ChainInput.contacts`): free to slide
anywhere inside its region, never left outside it."""

from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import FreeSpace
from scenarios import selected
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from kibends.bursts import OptimizerSettings
from kibends.energy import FloatArray, hinge_integrand
from kibends.neighbours import ChainInput
from kibends.obstacles import Obstacle, Obstacles
from kibends.optimize import (
    INSIDE_TOLERANCE,
    contact_stops,
    contacts_left,
    optimize_chains,
)
from kibends.results import ChainResult, Termination

SETTINGS = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)
PAD = box(-1.0, -1.0, 1.0, 1.0)
"""The region an end slides in: a pad already shrunk by its margin."""
OVER = box(4.0, 2.0, 6.0, 4.0)
"""The region a vertex partway along slides in."""
NEAR = 0.05
"""mm a sliding vertex ends from the edge of its region, at most, where the
length pulls it there: within the stop's standoff, inside."""
STEP = 1e-6
"""mm: the finite-difference step of the gradient check."""
GRADIENT_AGREEMENT = 1e-3
"""Relative difference allowed between the analytic gradient and the
finite difference at STEP, whose truncation error is far below it here."""
RESUMED_MOVE = 0.5
"""mm a resumed run moves a sliding vertex, at least, from where a run cut
short after one short burst left it (there 0.84 mm)."""
WEAK_STOP = 0.5
"""A clearance weight at which the stop cannot hold a vertex against the
length's pull: the length pulls with up to 2, the stop with this times the
track's width."""


def length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def sliding(chain: ChainInput, contacts: dict[int, BaseGeometry]) -> ChainInput:
    """`chain` with the vertices of `contacts` keeping their regions."""
    return replace(
        chain,
        contacts=tuple((v, Obstacles((), region)) for v, region in contacts.items()),
    )


def kept(result: ChainResult) -> FloatArray:
    """The vertices of `result` that keep a contact, in order."""
    return result.polyline[result.contact_of >= 0]


def end_in_pad() -> ChainInput:
    chain = selected("end", "A", FreeSpace(), (0.0, 0.0), (0.0, 4.0), (6.0, 4.0))
    return sliding(replace(chain, fixed=np.array([False, False, True])), {0: PAD})


def passing_over() -> ChainInput:
    chain = selected("over", "A", FreeSpace(), (0.0, 0.0), (5.0, 3.0), (10.0, 0.0))
    return sliding(replace(chain, fixed=np.array([True, False, True])), {1: OVER})


def test_an_end_in_a_pad_slides_toward_the_route_and_stays_inside() -> None:
    chain = end_in_pad()

    result = optimize_chains([chain], scenarios.PARAMS, SETTINGS)

    (got,) = result.chains
    assert result.termination is Termination.CONVERGED
    (end,) = kept(got)
    np.testing.assert_array_equal(end, got.polyline[0])
    assert PAD.covers(Point(end))
    # Pulled toward the far end, to the pad's corner nearest it.
    assert end == pytest.approx((1.0, 1.0), abs=NEAR)
    assert length(got.polyline) < got.before.length
    # Simplification keeps it where the run left it.
    assert got.fixed[0]


def test_a_vertex_kept_partway_along_stays_inside_and_the_rest_is_free() -> None:
    chain = passing_over()

    result = optimize_chains([chain], scenarios.PARAMS, SETTINGS)

    (got,) = result.chains
    (vertex,) = kept(got)
    assert OVER.covers(Point(vertex))
    # The chain would straighten along y = 0; the vertex holds it at the
    # region's near edge, and the rest runs straight to it.
    assert vertex[1] == pytest.approx(2.0, abs=NEAR)
    assert length(got.polyline) < got.before.length
    assert got.fixed.sum() == 3


def test_a_vertex_the_stop_cannot_hold_moves_back_inside() -> None:
    # Too weak to hold it, the stop lets each burst end with the vertex
    # outside, and the burst moves it back.
    weak = replace(scenarios.PARAMS, clearance_weight=WEAK_STOP)

    result = optimize_chains([passing_over()], weak, SETTINGS)

    (got,) = result.chains
    (vertex,) = kept(got)
    assert OVER.covers(Point(vertex))
    assert got.rejected_edges > 0


@pytest.mark.parametrize(
    ("at", "outside"),
    [
        ((5.0, 3.0), False),
        ((5.0, 2.0), False),
        ((5.0, 2.0 - INSIDE_TOLERANCE / 2), False),
        ((5.0, 2.0 - 2 * INSIDE_TOLERANCE), True),
        ((5.0, 1.0), True),
    ],
)
def test_a_vertex_beyond_the_tolerance_outside_its_contact_has_left_it(
    at: tuple[float, float], outside: bool
) -> None:
    points = np.array([(0.0, 0.0), at, at])
    contact_of = np.array([-1, 0, -1])

    left = contacts_left(points, contact_of, [Obstacles((), OVER)])

    assert left.tolist() == [False, outside, False]


def test_a_run_resumed_from_another_slides_its_vertices_again() -> None:
    chain = passing_over()
    short = replace(SETTINGS, max_iterations=1, burst_maxiter=1)
    first = optimize_chains([chain], scenarios.PARAMS, short)

    again = optimize_chains([chain], scenarios.PARAMS, SETTINGS, start=first)

    (before,) = kept(first.chains[0])
    (after,) = kept(again.chains[0])
    assert OVER.covers(Point(after))
    assert before[1] - after[1] >= RESUMED_MOVE


def test_a_frozen_sliding_vertex_is_pinned_and_keeps_no_contact() -> None:
    # Starting deep inside another net's clearance, the end cannot move.
    blocked = Obstacles([Obstacle("other", Point(0.3, 0.0), 0.5)])
    chain = replace(end_in_pad(), field=blocked)

    result = optimize_chains([chain], scenarios.PARAMS, SETTINGS)

    (got,) = result.chains
    np.testing.assert_array_equal(got.polyline[0], (0.0, 0.0))
    assert got.contact_of[0] < 0


def test_the_stop_s_gradient_matches_finite_differences() -> None:
    # Inside the standoff of the region's edge, where the stop rises.
    points = np.array([(0.99, 0.5), (3.0, 3.0)])
    contact_of = np.array([0, -1])
    within = [Obstacles((), PAD)]
    limit = hinge_integrand(scenarios.PARAMS)
    width = scenarios.PAIR_WIDTH

    value, gradient = contact_stops(points, contact_of, within, width, limit)

    assert value > 0.0
    for axis in range(2):
        moved = points.copy()
        moved[0, axis] += STEP
        ahead, _ = contact_stops(moved, contact_of, within, width, limit)
        difference = (ahead - value) / STEP
        assert difference == pytest.approx(gradient[0, axis], rel=GRADIENT_AGREEMENT)
    np.testing.assert_array_equal(gradient[1], (0.0, 0.0))
