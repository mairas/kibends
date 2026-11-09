import inspect
import itertools
import math
from typing import cast

import numpy as np
import pytest
import scenarios
from distance_fields import FloatArray, FreeSpace

from kibends.energy import MIN_EDGE_LENGTH, arc_lengths
from kibends.neighbours import (
    TOUCH_TOLERANCE,
    ChainInput,
    _complement,  # pyright: ignore[reportPrivateUsage]
    _leave_out,  # pyright: ignore[reportPrivateUsage]
    _vertex_at,  # pyright: ignore[reportPrivateUsage]
    kept_spans,
    neighbour_centrelines,
)
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


def test_a_junction_carried_away_is_found_at_the_chains_ends() -> None:
    # Both chains end at a via that started at (1, 0.2) and moved to
    # (5, 3); the first chain now runs up past where the junction was and
    # across to it, so one of its middle vertices is nearer that spot than
    # its end, and leaves it in another direction.
    junction = np.array([5.0, 3.0])
    own = chain("own", (1.0, 0.0), (1.0, 0.2))
    other = chain("other", (1.0, 0.2), (10.0, 0.0))
    own_now = np.concatenate(
        [
            np.linspace((1.0, 0.0), (1.0, 3.0), 7)[:-1],
            np.linspace((1.0, 3.0), junction, 9),
        ]
    )
    other_now = np.linspace(junction, (10.0, 0.0), 11)

    parts = neighbour_centrelines(own, other, other_now, REACH, own_now)

    kept = np.concatenate([part.points for part in parts])
    assert np.linalg.norm(kept - junction, axis=1).min() >= CLEAR - 1e-9


def _reference_heading(points: FloatArray, length: float) -> FloatArray:
    arc = arc_lengths(points)
    at = min(length, float(arc[-1]))
    ahead = np.array(
        [np.interp(at, arc, points[:, 0]), np.interp(at, arc, points[:, 1])]
    )
    step: FloatArray = ahead - points[0]
    return step / max(float(np.linalg.norm(step)), MIN_EDGE_LENGTH)


def _reference_departures(
    points: FloatArray, k: int, length: float
) -> list[tuple[bool, FloatArray]]:
    departures: list[tuple[bool, FloatArray]] = []
    if k < len(points) - 1:
        departures.append((True, _reference_heading(points[k:], length)))
    if k > 0:
        departures.append((False, _reference_heading(points[k::-1], length)))
    return departures


def reference_kept_spans(
    own: ChainInput,
    other: ChainInput,
    points: FloatArray,
    reach: float,
    own_points: FloatArray | None = None,
    margin: float = 0.0,
) -> list[tuple[float, float]] | None:
    """`kept_spans` as first written: every touch pair on its own, each
    heading from the chain's whole arc length.

    It pins the refactor that finds each pin's departures once by binary
    search; the next intended change to the spans replaces or removes it."""
    me, them = own.centreline, other.centreline
    if them.net != me.net:
        return None
    pinned_mine = me.points[own.fixed]
    pinned_theirs = them.points[other.fixed]
    overlap = (me.width + them.width) / 2 + TOUCH_TOLERANCE
    gaps = np.linalg.norm(pinned_mine[:, None] - pinned_theirs[None], axis=2)
    touches = list(zip(*np.nonzero(gaps <= overlap), strict=True))
    if not touches:
        return None
    mine = me.points if own_points is None else own_points
    clear = (me.width + them.width) / 2 + reach + margin
    arc = arc_lengths(points)
    left_out: list[tuple[float, float]] = []
    for a, b in touches:
        own_pin = _vertex_at(me.points, mine, pinned_mine[a])
        other_pin = _vertex_at(them.points, points, pinned_theirs[b])
        own_headings = [h for _, h in _reference_departures(mine, own_pin, clear)]
        offset: FloatArray = points[other_pin] - mine[own_pin]
        along = float(arc[other_pin])
        for forward, heading in _reference_departures(points, other_pin, clear):
            length = _leave_out(own_headings, offset, heading, clear)
            left_out.append(
                (along, along + length) if forward else (along - length, along)
            )
    return _complement(left_out, float(arc[-1]))


def _scenario_chains() -> list[tuple[str, tuple[ChainInput, ...]]]:
    """The chains of every scenario that takes no arguments."""
    found: list[tuple[str, tuple[ChainInput, ...]]] = []
    for name, make in inspect.getmembers(scenarios, inspect.isfunction):
        if make.__module__ != scenarios.__name__ or any(
            p.default is inspect.Parameter.empty
            for p in inspect.signature(make).parameters.values()
        ):
            continue
        chains: object = getattr(make(), "chains", None)
        if isinstance(chains, tuple) and chains:
            found.append((name, cast(tuple[ChainInput, ...], chains)))
    return found


def _jittered(chain: ChainInput, rng: np.random.Generator) -> FloatArray:
    """The chain's polyline with its free vertices moved a little."""
    points = chain.centreline.points.copy()
    free = ~chain.fixed
    points[free] += rng.normal(0.0, 0.05, (int(free.sum()), 2))
    return points


def _resampled(chain: ChainInput, rng: np.random.Generator) -> FloatArray:
    """The chain's polyline with a vertex added in the middle of each edge,
    as resampling adds them, and every vertex but its pins moved a little."""
    points = chain.centreline.points
    resampled = np.empty((2 * len(points) - 1, 2))
    resampled[::2] = points
    resampled[1::2] = (points[:-1] + points[1:]) / 2
    free = np.ones(len(resampled), dtype=bool)
    free[::2] = ~chain.fixed
    resampled[free] += rng.normal(0.0, 0.05, (int(free.sum()), 2))
    return resampled


END_GAP = 0.1
"""How far the bar's inner pins lie from its ends, closer than `clear`."""


def _end_pins() -> tuple[ChainInput, ChainInput]:
    """A bar pinned `END_GAP` from each end, and an arch whose ends touch it
    there."""
    bar = chain("bar", (0.0, 0.0), (END_GAP, 0.0), (LEG - END_GAP, 0.0), (LEG, 0.0))
    bar.fixed[[1, 2]] = True
    arch = chain(
        "arch",
        (END_GAP, 0.1),
        (END_GAP, 1.0),
        (LEG - END_GAP, 1.0),
        (LEG - END_GAP, 0.1),
    )
    return bar, arch


def _clustered(
    name: str, wave: float, pins: slice, rng: np.random.Generator
) -> ChainInput:
    """A chain waving `wave` about y = 0, crossing it at x = π, with a run of
    pinned vertices 0.05 mm apart."""
    x = np.linspace(0.0, 10.0, 201)
    y = wave * np.sin(x) + rng.normal(0.0, 0.01, len(x))
    points = np.column_stack([x, y])
    fixed = np.zeros(len(points), dtype=bool)
    fixed[[0, -1]] = True
    fixed[pins] = True
    return ChainInput(
        ChainCentreline(name, "N", WIDTH, 0.2, points), "F.Cu", fixed, FreeSpace()
    )


def _clustered_pair() -> tuple[ChainInput, ChainInput]:
    rng = np.random.default_rng(64)
    return (
        _clustered("own", 0.3, slice(55, 71), rng),
        _clustered("other", -0.3, slice(52, 74), rng),
    )


def _scenario_pairs() -> list[tuple[str, ChainInput, ChainInput]]:
    """Every ordered pair of chains of one net in a scenario."""
    return [
        (f"{name}-{a}-{b}", own, other)
        for name, chains in _scenario_chains()
        for (a, own), (b, other) in itertools.permutations(enumerate(chains), 2)
        if own.centreline.net == other.centreline.net
    ]


def _same_net_pairs() -> list[tuple[str, ChainInput, ChainInput]]:
    own, other = _clustered_pair()
    bar, arch = _end_pins()
    return [
        *_scenario_pairs(),
        ("clustered", own, other),
        ("clustered-back", other, own),
        ("end-pins", arch, bar),
        ("end-pins-back", bar, arch),
    ]


def _assert_same_spans(
    got: list[tuple[float, float]] | None, want: list[tuple[float, float]] | None
) -> None:
    if want is None:
        assert got is None
        return
    assert got is not None
    assert len(got) == len(want)
    assert np.array(got) == pytest.approx(np.array(want), abs=1e-9)


@pytest.mark.parametrize(
    ("own", "other"),
    [pytest.param(own, other, id=name) for name, own, other in _same_net_pairs()],
)
@pytest.mark.parametrize("margin", [0.0, 0.2])
def test_kept_spans_match_the_reference(
    own: ChainInput, other: ChainInput, margin: float
) -> None:
    rng = np.random.default_rng(7)
    starts = (own.centreline.points, other.centreline.points)
    moved = (_jittered(own, rng), _jittered(other, rng))
    resampled = (_resampled(own, rng), _resampled(other, rng))
    for own_points, points in (starts, moved, resampled):
        _assert_same_spans(
            kept_spans(own, other, points, REACH, own_points, margin),
            reference_kept_spans(own, other, points, REACH, own_points, margin),
        )


def test_the_clustered_pair_is_kept_on_both_sides_of_the_crossing() -> None:
    own, other = _clustered_pair()
    spans = kept_spans(own, other, other.centreline.points, REACH)
    assert spans is not None
    assert len(spans) == 2


def test_every_scenario_pair_of_one_net_is_compared() -> None:
    assert len(_scenario_pairs()) >= 38


@pytest.mark.parametrize("margin", [0.0, 0.2])
def test_the_end_pins_lie_within_clear_of_both_ends(margin: float) -> None:
    # Departures from them reach past the bar's start backwards and past
    # its end forwards, so both ends of the heading's range are clamped.
    bar, _ = _end_pins()
    clear = WIDTH + REACH + margin
    arc = arc_lengths(bar.centreline.points)
    assert arc[1] - clear < 0.0
    assert arc[2] + clear > arc[-1]
