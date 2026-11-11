# pyright: reportPrivateUsage=false
"""What a hold puts back where it started: `optimize._Problem.failing`,
each of its criteria alone, on states laid out by hand."""

from collections.abc import Sequence

import numpy as np
import scenarios
from distance_fields import BoolArray, Box, FloatArray, FreeSpace
from scenarios import ViasScenario, selected

import kibends.optimize as optimize_module
from kibends.bursts import OptimizerSettings
from kibends.neighbours import ChainInput
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE
from kibends.via_moves import MovingVias
from kibends.vias import ViaInput

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)

BLOCK = Box(5.0, 2.0, 1.0, 0.5)
"""Static copper above the middle of a chain along y = 0, inflated."""
LIFT = 2.0
"""mm that a chain's free vertices rise in a state laid out by hand."""


def problem(
    chains: Sequence[ChainInput], vias: Sequence[ViaInput] = ()
) -> optimize_module._Problem:
    """The one problem of `chains` and `vias`."""
    moving = MovingVias(chains, vias, HOLE_CLEARANCE, HOLE_TO_HOLE)
    (placed,) = optimize_module._problems(chains, [], moving, scenarios.PARAMS, COARSE)
    return placed.problem


def failing(
    at: optimize_module._Problem,
    points: Sequence[FloatArray] | None = None,
    positions: Sequence[FloatArray] | None = None,
) -> BoolArray:
    """`failing` of `at` with the members at `points` and the vias at
    `positions`, by default where the problem starts."""
    state = optimize_module._State(
        list(at.points if points is None else points),
        at.masks,
        at.contact_of,
        list(at.positions if positions is None else positions),
    )
    start = at.entered(at.points, at.positions, at.seen)
    return at.failing(state, at.seeing(state.points), start)


def lifted(points: FloatArray) -> FloatArray:
    """`points` with every vertex but the ends `LIFT` higher."""
    moved = points.copy()
    moved[1:-1, 1] += LIFT
    return moved


def test_nothing_fails_where_the_problem_starts() -> None:
    at = problem(scenarios.via_pair().chains, scenarios.via_pair().vias)

    assert not failing(at).any()


def test_a_member_with_more_violations_fails() -> None:
    at = problem((selected("A", "A", BLOCK, (0.0, 0.0), (10.0, 0.0)),))

    # Across static copper, which no other member enters.
    assert failing(at, [lifted(at.points[0])]).tolist() == [True]


ANCHOR = (10.0, 0.0)


def test_a_member_entered_at_its_anchor_fails_with_what_entered_it() -> None:
    # A run of vertices inside clearance that holds an anchor is no
    # violation of its own: it cannot move away.
    at = problem(
        (
            selected("A", "A", FreeSpace(), (0.0, 0.0), ANCHOR),
            selected("B", "B", FreeSpace(), (8.0, -LIFT), (12.0, -LIFT)),
        )
    )

    assert failing(at, [at.points[0], lifted(at.points[1])]).tolist() == [True, True]


def via_scenario(*, one_net: bool) -> ViasScenario:
    """`scenarios.via_pair`, its two routes of one net if `one_net`."""
    if not one_net:
        return scenarios.via_pair()
    lower, upper = (5.0, -2.0), (5.0, scenarios.PAIR_LEVEL + 2.0)
    level = scenarios.PAIR_LEVEL
    return ViasScenario(
        "via-pair-one-net",
        (
            *scenarios.route("A", lower),
            *scenarios.route("A", upper, (0.0, level), (10.0, level)),
        ),
        (
            scenarios.joining(scenarios.through_via("a", "A", lower), 0),
            scenarios.joining(scenarios.through_via("b", "A", upper), 2),
        ),
    )


def vias_failing(scenario: ViasScenario, b_at: tuple[float, float]) -> list[bool]:
    """Whether each via of `scenario` fails with the second at `b_at`."""
    at = problem(scenario.chains, scenario.vias)
    moved = [at.positions[0], np.array(b_at, dtype=float)]
    return failing(at, positions=moved)[len(at.members) :].tolist()


VIA_GAP = 0.7
"""mm between the vias' centres, below the copper's `VIA_TO_VIA`, beyond
`VIA_TO_CHAIN` from the end of the other's chains."""
STOPPED_GAP = 0.4
"""mm between the centres of two vias of one net, inside how far apart their
holes keep."""


def test_a_via_entering_another_net_s_via_fails_with_it() -> None:
    assert VIA_GAP < scenarios.VIA_TO_VIA
    assert VIA_GAP > scenarios.VIA_TO_CHAIN

    assert vias_failing(via_scenario(one_net=False), (5.0, -2.0 - VIA_GAP)) == [
        True,
        True,
    ]


def test_a_via_inside_its_stops_fails() -> None:
    assert vias_failing(via_scenario(one_net=True), (5.0, -2.0 - STOPPED_GAP))[1]


def test_a_via_inside_its_static_copper_fails() -> None:
    scenario = scenarios.via_hemmed()
    at = problem(scenario.chains, scenario.vias)
    track = scenarios.HEMMING_TRACK

    across = failing(at, positions=[np.array([track.cx, track.cy])])

    assert across[len(at.members) :].tolist() == [True]
