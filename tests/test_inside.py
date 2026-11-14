"""The start check: a run refuses selected copper that starts inside a
clearance as drawn, naming each place (`inside`)."""

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
import scenarios
from conftest import (
    NM_ROUNDING,
    FakeBoard,
    kipy_pad,
    kipy_rectangle,
    kipy_via,
    layer_snapshot,
    refused,
)
from distance_fields import Disc, FreeSpace
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from scenarios import PAIR_CLEARANCE, PAIR_WIDTH, selected
from shapely.geometry import LineString, Point, Polygon, box

from kibends.bursts import DEFAULT_SETTINGS, INSIDE_TOLERANCE
from kibends.chain import NM_PER_MM, Point2
from kibends.coords import UserFrame
from kibends.inside import (
    INSIDE_MESSAGE,
    START_TOLERANCE,
    ArcChords,
    Chord,
    InsideClearanceError,
    Violation,
    arc_chords,
    sagitta,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    EDGE_CLEARANCE,
    HOLE_TO_HOLE,
    Obstacle,
    Obstacles,
    build_obstacles,
    build_pair_obstacles,
)
from kibends.optimize import optimize_chains
from kibends.pair_blocks import PairInput
from kibends.results import RunResult, Termination
from kibends.snapshot import Arc
from kibends.vias import ViaInput

SETTINGS = DEFAULT_SETTINGS
PAD_AT = (10.0, 0.0)
"""Where the chain ends, pinned on a pad."""
BESIDE_GAP = 0.15
"""How far the pad beside the route's middle is from its copper, mm: inside
the clearance."""


def ending_on_pad(net: str) -> ChainInput:
    """A chain of SIG ending on a pad of `net`."""
    snap = layer_snapshot(FakeBoard(pads=[kipy_pad("p", net, *PAD_AT)]), "F.Cu")
    field = build_obstacles(snap, "SIG", PAIR_WIDTH)
    return selected("chain", "SIG", field, (0.0, 0.0), (5.0, 2.0), PAD_AT)


def test_a_chain_pinned_on_a_pad_of_its_net_starts() -> None:
    result = optimize_chains([ending_on_pad("SIG")], scenarios.PARAMS, SETTINGS)

    assert result.termination is Termination.CONVERGED


def test_a_pad_of_another_net_beside_a_route_is_refused_by_name() -> None:
    half = 0.25
    """The pads' half size, as `kipy_pad` makes them."""
    y = PAIR_WIDTH / 2 + BESIDE_GAP + half
    snap = layer_snapshot(FakeBoard(pads=[kipy_pad("q", "GND", 5.0, y)]), "F.Cu")
    field = build_obstacles(snap, "SIG", PAIR_WIDTH)
    chain = selected("chain", "SIG", field, (0.0, 0.0), (10.0, 0.0))

    (violation,) = refused(lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS))

    assert (violation.subject, violation.obstacle) == ("chain", "pad q")
    assert violation.required == pytest.approx(snap.clearance("SIG"))
    assert violation.distance == pytest.approx(BESIDE_GAP)
    assert violation.at[1] == 0.0
    # Along the pad's flat side the route is as deep everywhere.
    assert abs(violation.at[0] - 5.0) <= half


RADIUS = 1.0
"""mm, of the arc hugging a round obstacle."""


LEAD = 1.0
"""mm of the radial leads from the chain's pinned ends to its arc."""


def round_arc() -> tuple[Arc, list[Point2]]:
    """A quarter arc of RADIUS about the origin, and the points of its
    chords."""
    r = round(RADIUS * NM_PER_MM)
    diagonal = round(RADIUS * NM_PER_MM / math.sqrt(2))
    arc = Arc("a", (r, 0), (diagonal, diagonal), (0, r), r // 5, "F.Cu", "SIG")
    ends = [arc.segments()[0].start, *(s.end for s in arc.segments())]
    return arc, [(x / NM_PER_MM, y / NM_PER_MM) for x, y in ends]


def hugging(extra: float) -> tuple[ChainInput, Arc]:
    """A quarter arc of RADIUS about the origin, as its chords, hugging a
    round obstacle there whose clearance reaches `extra` beyond the arc,
    between radial leads that end, pinned, outside it."""
    arc, points = round_arc()
    leads = [(RADIUS + LEAD, 0.0), *points, (0.0, RADIUS + LEAD)]
    disc = Obstacles([Obstacle("disc", Point(0.0, 0.0), RADIUS + extra)])
    return selected("arc", "SIG", disc, *leads), arc


def test_an_arc_s_chords_inside_its_obstacle_by_their_sagitta_are_refused() -> None:
    chain, _ = hugging(0.0)

    assert refused(lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS))


def test_an_arc_s_chords_within_their_sagitta_start() -> None:
    chain, arc = hugging(0.0)

    result = optimize_chains(
        [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
    )

    assert result.termination is not Termination.ABNORMAL


def test_an_arc_inside_its_obstacle_beyond_its_sagitta_is_refused() -> None:
    _, drawn = hugging(0.0)
    deepest = max(c.sagitta for c in arc_chords([drawn]).values())
    chain, arc = hugging(10 * deepest)

    (violation,) = refused(
        lambda: optimize_chains(
            [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
        )
    )

    assert violation.depth > deepest


def test_the_chords_sagitta_is_how_far_they_sit_inside_the_arc() -> None:
    chain, arc = hugging(0.0)

    chords = arc_chords([arc])

    assert len(chords) == len(chain.centreline.points) - 3
    for (a, b), chord in chords.items():
        middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        assert chord.sagitta == pytest.approx(
            RADIUS - math.hypot(*middle), abs=NM_ROUNDING
        )
        assert chord.centre == pytest.approx((0.0, 0.0), abs=NM_ROUNDING)


def test_the_refusal_prints_each_place_in_the_run_s_frame() -> None:
    near = Violation("chain", "pad q", (5.0, 1.0), 0.05, PAIR_CLEARANCE)
    over = Violation("chain", "track t", (6.0, 1.0), 0.3, PAIR_CLEARANCE)
    deep = Violation("via v", "via keepout", (2.0, 3.0), 0.1, None)
    frame = UserFrame((1.0, 2.0))

    error = InsideClearanceError([near, over, deep]).in_frame(frame)

    assert str(error).splitlines() == [
        INSIDE_MESSAGE,
        "  chain is 0.150 mm from pad q at (4.000, -1.000) mm, 0.200 mm required",
        "  chain overlaps track t by 0.100 mm at (5.000, -1.000) mm, 0.200 mm required",
        "  via v is 0.100 mm inside its clearance of via keepout at (1.000, 1.000) mm",
    ]
    assert error.violations == (near, over, deep)
    assert near.depth > INSIDE_TOLERANCE


SECTION_OUTER = scenarios.DP_PITCH / 2 + PAIR_WIDTH / 2
"""From a straight section's centreline to its tracks' outer copper, mm."""
VIA_BEYOND = SECTION_OUTER + PAIR_CLEARANCE + scenarios.VIA_DIAMETER / 2
"""How far a via's centre beside a straight section is at its clearance."""
NEAR = 0.001
"""mm the via beside the section stands beyond or inside its clearance."""


def beside_section(off: float) -> Callable[[], RunResult]:
    """A straight coupled section and a moving via of another net beside its
    middle, `off` beyond its clearance of the section's tracks."""
    section = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    via = ViaInput(
        scenarios.through_via("v", "V", (5.0, VIA_BEYOND + off)),
        (),
        FreeSpace(),
        PAIR_CLEARANCE,
    )
    return lambda: optimize_chains(
        [], scenarios.PARAMS, SETTINGS, vias=[via], pairs=[section]
    )


def test_a_via_just_beyond_a_section_s_clearance_starts() -> None:
    # The optimizer's stand-in for the section, wide enough to cover its
    # mitred corners while it moves, reaches the via; the tracks do not.
    result = beside_section(NEAR)()

    assert result.termination is not Termination.ABNORMAL


def test_a_via_just_inside_a_section_s_clearance_is_refused() -> None:
    (violation,) = refused(beside_section(-NEAR))

    assert (violation.subject, violation.obstacle) == (
        "via v",
        "the section of P and N",
    )
    assert violation.required == pytest.approx(PAIR_CLEARANCE)
    assert violation.depth == pytest.approx(NEAR)


ROUND_CORNER = 3.0
"""mm, the radius of a pair's centreline around a corner."""
CORNER_STEPS = 18
"""Chords of the corner, 5 degrees each, as KiCad's arcs are segmented."""


def arc_cornered() -> tuple[PairInput, ArcChords]:
    """A pair whose centreline rounds a quarter corner of ROUND_CORNER as
    chords about a round obstacle whose clearance it touches there, and the
    sagittas of its tracks' chords."""
    corner = [
        (ROUND_CORNER * math.cos(t), ROUND_CORNER * math.sin(t))
        for t in (math.pi / 2 * k / CORNER_STEPS for k in range(CORNER_STEPS + 1))
    ]
    centre = [(ROUND_CORNER + 1.0, 0.0), *corner, (0.0, ROUND_CORNER + 1.0)]
    chords: dict[tuple[Point2, Point2], Chord] = {}
    for track in scenarios.tracks(*centre):
        vertices = [(float(x), float(y)) for x, y in track[1 : CORNER_STEPS + 2]]
        for a, b in pairwise(vertices):
            chord = sagitta(math.hypot(*a), math.dist(a, b))
            chords[(a, b)] = Chord(chord, (0.0, 0.0))
    return scenarios.coupled(Disc(0.0, 0.0, ROUND_CORNER), *centre), chords


def test_an_arc_cornered_pair_at_clearance_starts_by_its_chords() -> None:
    section, chords = arc_cornered()

    assert refused(
        lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])
    )
    result = optimize_chains(
        [], scenarios.PARAMS, SETTINGS, pairs=[section], arc_chords=chords
    )
    assert result.termination is not Termination.ABNORMAL


def test_a_result_resting_at_clearance_rounded_to_the_grid_runs_again() -> None:
    # Pressed against each other's clearance, which the weak hinge cannot
    # hold them off, and rounded to KiCad's grid as KiCad holds them.
    scenario = scenarios.squeeze()
    params = replace(scenario.params, clearance_weight=0.1)
    first = optimize_chains(scenario.chains, params, SETTINGS)
    again = [
        replace(
            given,
            centreline=replace(
                given.centreline,
                points=np.round(chain.polyline * NM_PER_MM) / NM_PER_MM,
            ),
            fixed=chain.fixed,
        )
        for given, chain in zip(scenario.chains, first.chains, strict=True)
    ]

    second = optimize_chains(again, params, SETTINGS)

    assert second.termination is not Termination.ABNORMAL


WIDE_RADIUS = 20.0
"""mm, of a gently rounded track, whose 5 degree chords sit about 17 µm
inside it."""
PAD_REACH = 0.5
"""mm, how far a round obstacle's clearance reaches from its centre."""


def wide_arc(radius: float = WIDE_RADIUS) -> tuple[Arc, list[Point2]]:
    """A quarter arc of `radius` about the origin, and the points of its
    chords."""
    r = round(radius * NM_PER_MM)
    d = round(radius * NM_PER_MM / math.sqrt(2))
    arc = Arc("a", (r, 0), (d, d), (0, r), r // 100, "F.Cu", "SIG")
    ends = [arc.segments()[0].start, *(s.end for s in arc.segments())]
    return arc, [(x / NM_PER_MM, y / NM_PER_MM) for x, y in ends]


ARC_DEPTH = 0.9
"""How far into the clearance the arc reaches, as a fraction of its chords'
sagitta: the chords alone are inside on the concave side, clear on the
convex side."""


def arc_direction(
    points: Sequence[Point2], *, at_vertex: bool = False, along: float = 0.0
) -> Point2:
    """The unit direction from the arc's centre through a middle chord's
    middle, turned `along` its half angle toward its end, or through its
    first vertex."""
    k = len(points) // 2
    a, b = points[k], points[k + 1]
    if at_vertex:
        angle = math.atan2(a[1], a[0])
    else:
        first, last = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
        angle = (first + last) / 2 + along * (last - first) / 2
    return (math.cos(angle), math.sin(angle))


def deepest_sagitta() -> float:
    arc, _ = wide_arc()
    return max(c.sagitta for c in arc_chords([arc]).values())


def beyond_the_arc(
    points: Sequence[Point2],
    reach: float,
    depth: float,
    *,
    outside: bool,
    at_vertex: bool = False,
) -> Point2:
    """Where the centre of what keeps `reach` from the arc's centreline
    lies for the arc to reach `depth` into that clearance, outside the arc
    or inside it."""
    ux, uy = arc_direction(points, at_vertex=at_vertex)
    at = WIDE_RADIUS + (reach - depth if outside else depth - reach)
    return (at * ux, at * uy)


def near_the_arc(
    depth: float, *, outside: bool, at_vertex: bool = False
) -> Callable[[], RunResult]:
    arc, points = wide_arc()
    at = beyond_the_arc(points, PAD_REACH, depth, outside=outside, at_vertex=at_vertex)
    field = Obstacles([Obstacle("pad", Point(at), PAD_REACH)])
    chain = selected("arc", "SIG", field, *points)
    chords = arc_chords([arc])
    return lambda: optimize_chains(
        [chain], scenarios.PARAMS, SETTINGS, arc_chords=chords
    )


def test_an_arc_inside_a_clearance_at_its_vertex_is_refused_by_the_chords() -> None:
    # On the concave side the chords beside the vertex reach nearer than
    # the arc, which is inside only at the vertex: a place there is judged
    # as straight copper, as deep as the chords are, no shallower than the
    # arc.
    depth = ARC_DEPTH * deepest_sagitta()
    _, points = wide_arc()
    at = beyond_the_arc(points, PAD_REACH, depth, outside=False, at_vertex=True)
    chords = LineString(points).distance(Point(at))

    found = refused(near_the_arc(depth, outside=False, at_vertex=True))

    assert {v.obstacle for v in found} == {"pad"}
    deepest = max(v.depth for v in found)
    assert depth - START_TOLERANCE <= deepest <= PAD_REACH - chords + START_TOLERANCE


def test_an_arc_at_a_clearance_on_its_concave_side_starts() -> None:
    result = near_the_arc(0.0, outside=False)()

    assert result.termination is not Termination.ABNORMAL


def convex_pad(depth: float) -> tuple[list[ChainInput], list[ViaInput]]:
    """The arc and a pad outside it, its clearance reaching `depth` into the
    arc at a chord's middle."""
    _, points = wide_arc()
    at = beyond_the_arc(points, PAD_REACH, depth, outside=True)
    field = Obstacles([Obstacle("pad", Point(at), PAD_REACH)])
    return [selected("arc", "SIG", field, *points)], []


def convex_chain(depth: float) -> tuple[list[ChainInput], list[ViaInput]]:
    """The arc and a chain of another net outside it, its tip reaching
    `depth` into their clearance of the arc at a chord's middle."""
    _, points = wide_arc()
    tip = beyond_the_arc(points, scenarios.PAIR_INFLATION, depth, outside=True)
    ux, uy = arc_direction(points)
    tail = (tip[0] + ux, tip[1] + uy)
    return [
        selected("arc", "SIG", FreeSpace(), *points),
        selected("outer", "OTHER", FreeSpace(), tail, tip),
    ], []


def convex_via(depth: float) -> tuple[list[ChainInput], list[ViaInput]]:
    """The arc and a moving via of another net outside it, reaching `depth`
    into their clearance of the arc at a chord's middle."""
    _, points = wide_arc()
    at = beyond_the_arc(points, scenarios.VIA_TO_CHAIN, depth, outside=True)
    via = ViaInput(scenarios.through_via("v", "V", at), (), FreeSpace(), PAIR_CLEARANCE)
    return [selected("arc", "SIG", FreeSpace(), *points)], [via]


@pytest.mark.parametrize(
    "outside", [convex_pad, convex_chain, convex_via], ids=["pad", "chain", "via"]
)
def test_an_arc_between_clear_chords_on_its_convex_side_starts(
    outside: Callable[[float], tuple[list[ChainInput], list[ViaInput]]],
) -> None:
    # The chords are judged as they are: they replace the arc in the
    # result, and an unchanged chain leaves the drawn arc to KiCad's DRC.
    arc, _ = wide_arc()
    chains, vias = outside(ARC_DEPTH * deepest_sagitta())

    result = optimize_chains(
        chains, scenarios.PARAMS, SETTINGS, vias=vias, arc_chords=arc_chords([arc])
    )

    assert result.termination is not Termination.ABNORMAL


CONCAVE_ALONG = 0.02
"""How far from a chord's middle toward its end the via inside the arc
lies, as a fraction of the chord's half angle: between the chord's ends,
where the chord's allowance, the arc's offset from it square to the chord,
exceeds what the via needs by under START_TOLERANCE."""


def inside_the_arc(radius: float, depth: float) -> Callable[[], RunResult]:
    """An arc of `radius` and a moving via of another net inside it, off a
    chord's middle, reaching `depth` into their clearance of the arc."""
    arc, points = wide_arc(radius)
    ux, uy = arc_direction(points, along=CONCAVE_ALONG)
    at = radius - scenarios.VIA_TO_CHAIN + depth
    via = ViaInput(
        scenarios.through_via("v", "V", (at * ux, at * uy)),
        (),
        FreeSpace(),
        PAIR_CLEARANCE,
    )
    chain = selected("arc", "SIG", FreeSpace(), *points)
    return lambda: optimize_chains(
        [chain], scenarios.PARAMS, SETTINGS, vias=[via], arc_chords=arc_chords([arc])
    )


@pytest.mark.parametrize("radius", [50.0, 100.0])
def test_a_via_at_clearance_inside_a_wide_arc_starts(radius: float) -> None:
    # Its chord is inside the clearance by nearly the chord's sagitta.
    result = inside_the_arc(radius, 0.0)()

    assert result.termination is not Termination.ABNORMAL


@pytest.mark.parametrize("radius", [50.0, 100.0])
def test_a_via_just_inside_a_wide_arc_s_clearance_is_refused(radius: float) -> None:
    (violation,) = refused(inside_the_arc(radius, 2 * START_TOLERANCE))

    assert (violation.subject, violation.obstacle) == ("via v", "arc")
    assert violation.depth > START_TOLERANCE


OWN_NET_GAP = 0.1
"""mm between a pad and a section's track beside it: inside the clearance,
not touching."""


def section_beside_pad(net: str) -> Callable[[], RunResult]:
    """A straight section P/N and a pad of `net` OWN_NET_GAP beyond track P's
    copper."""
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    p_y = float(straight.a.points[0][1])
    away = math.copysign(1.0, p_y)
    half = 0.25
    """The pads' half size, as `kipy_pad` makes them."""
    y = p_y + away * (PAIR_WIDTH / 2 + OWN_NET_GAP + half)
    snap = layer_snapshot(FakeBoard(pads=[kipy_pad("q", net, 5.0, y)]), "F.Cu")
    field = build_pair_obstacles(snap, straight.a, straight.b, straight.gap)
    section = replace(straight, field=field)
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])


def test_a_section_beside_a_pad_of_its_track_s_net_starts() -> None:
    result = section_beside_pad("P")()

    assert result.termination is not Termination.ABNORMAL


@pytest.mark.parametrize("net", ["N", "GND"])
def test_a_section_beside_a_pad_of_another_net_is_refused(net: str) -> None:
    (violation,) = refused(section_beside_pad(net))

    assert violation.obstacle == "pad q"
    assert violation.distance == pytest.approx(OWN_NET_GAP)


VIA_GAP = 0.1
"""mm between two moving vias' rings: inside their clearance."""


def two_vias(net: str, apart: float) -> Callable[[], RunResult]:
    """A stitching via of net A and one of `net`, `apart` between centres."""
    vias = [
        ViaInput(
            scenarios.through_via(name, of, (5.0, y)), (), FreeSpace(), PAIR_CLEARANCE
        )
        for name, of, y in (("a", "A", 0.0), ("b", net, apart))
    ]
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, vias=vias)


def test_two_moving_vias_inside_each_other_s_clearance_are_refused() -> None:
    apart = scenarios.VIA_DIAMETER + VIA_GAP

    (violation,) = refused(two_vias("B", apart))

    assert (violation.subject, violation.obstacle) == ("via a", "via b")
    assert violation.required == pytest.approx(PAIR_CLEARANCE)
    assert violation.distance == pytest.approx(VIA_GAP)


def test_two_vias_of_one_net_closer_than_hole_to_hole_are_refused() -> None:
    # Of one net their copper may touch; only hole-to-hole holds them apart.
    apart = scenarios.VIA_DRILL + HOLE_TO_HOLE - VIA_GAP

    (violation,) = refused(two_vias("A", apart))

    assert violation.required is None
    assert violation.depth == pytest.approx(VIA_GAP)


def test_a_chain_at_clearance_on_an_arc_s_concave_side_starts() -> None:
    # A chain pinned inside the arc, its tip at their clearance from the
    # arc where a chord sits nearer by its sagitta: between two chains, as
    # against static copper, the chord's slack lets it start.
    arc, points = wide_arc()
    a, b = points[len(points) // 2], points[len(points) // 2 + 1]
    angle = math.atan2(a[1] + b[1], a[0] + b[0])
    unit = (math.cos(angle), math.sin(angle))
    tip_at = WIDE_RADIUS - scenarios.PAIR_INFLATION
    tip = (tip_at * unit[0], tip_at * unit[1])
    tail = ((tip_at - 1.0) * unit[0], (tip_at - 1.0) * unit[1])
    chains = [
        selected("arc", "SIG", FreeSpace(), *points),
        selected("inner", "OTHER", FreeSpace(), tail, tip),
    ]

    assert refused(lambda: optimize_chains(chains, scenarios.PARAMS, SETTINGS))
    result = optimize_chains(
        chains, scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
    )
    assert result.termination is not Termination.ABNORMAL


EDGE_DIP = 2.0
"""mm a section's ends lie below its middle, which rises toward the edge."""
EDGE_DEPTH = 0.1
"""mm the section's middle starts inside the board edge's clearance."""


@pytest.mark.parametrize("pad_elsewhere", [False, True], ids=["alone", "beside-pad"])
def test_a_section_inside_the_board_edge_s_clearance_is_refused(
    pad_elsewhere: bool,
) -> None:
    # A pad far away gives the field an item, which must not stand for the
    # edge's clearance.
    centre = [(0.0, -EDGE_DIP), (5.0, 0.0), (10.0, -EDGE_DIP)]
    given = scenarios.coupled(FreeSpace(), *centre)
    top = max(float(np.max(t.points[:, 1])) for t in (given.a, given.b))
    edge_y = top + PAIR_WIDTH / 2 + EDGE_CLEARANCE - EDGE_DEPTH
    pads = [kipy_pad("far", "GND", 5.0, -6.0)] if pad_elsewhere else []
    outline = kipy_rectangle((-5.0, -10.0), (15.0, edge_y), layer=BL.BL_Edge_Cuts)
    snap = layer_snapshot(FakeBoard(pads=pads, shapes=[outline]), "F.Cu")
    field = build_pair_obstacles(snap, given.a, given.b, given.gap)
    section = replace(given, field=field)

    (violation,) = refused(
        lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])
    )

    assert violation.obstacle == "the board edge"
    assert violation.depth > 0.0


HOLE_GAP = 0.05
"""mm a via's ring stays clear of the section's track beside it, its hole
inside the hole clearance."""


def section_beside_via(net: str) -> Callable[[], RunResult]:
    """A straight section P/N and an unselected via of `net` beside track P,
    its ring HOLE_GAP clear of the track's copper, its hole inside the
    track's hole clearance."""
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    p_y = float(straight.a.points[0][1])
    away = math.copysign(1.0, p_y)
    ring = scenarios.VIA_DIAMETER / 2
    y = p_y + away * (PAIR_WIDTH / 2 + ring + HOLE_GAP)
    via = kipy_via("h", net, 5.0, y)
    snap = layer_snapshot(FakeBoard(vias=[via]), "F.Cu")
    field = build_pair_obstacles(snap, straight.a, straight.b, straight.gap)
    section = replace(straight, field=field)
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])


def test_a_section_beside_a_via_of_its_track_s_net_starts() -> None:
    # Its ring and its hole are of the track's net: no clearance applies.
    result = section_beside_via("P")()

    assert result.termination is not Termination.ABNORMAL


REAL_DEPTH = 0.5
"""How far into the clearance the second obstacle reaches the outer track,
as a fraction of the inner track's chords' sagitta, at a vertex."""


def test_a_section_inside_one_obstacle_is_refused_beside_one_its_chords_may_be_in() -> (
    None
):
    # The pad round the corner's centre keeps the inner track's vertices,
    # mitred inward of the centreline's circle, at its clearance, its
    # chords inside it by their sagitta on its concave side: the deeper,
    # but allowed. The point beside the outer track reaches it at a vertex,
    # less deep than the centre reaches the chords but truly inside.
    section, chords = arc_cornered()
    inner = min(
        (track for track in (section.a, section.b)),
        key=lambda t: float(np.hypot(*t.points[2])),
    )
    outer = section.b if inner is section.a else section.a
    extent = (section.gap + section.a.width) / 2 + section.a.width / 2
    k = CORNER_STEPS // 2 + 1
    a, b = (float(x) for x in inner.points[k]), (float(x) for x in inner.points[k + 1])
    a2, b2 = tuple(a), tuple(b)
    inner_sagitta = sagitta(math.hypot(*a2), math.dist(a2, b2))
    """How far the inner track's chords at the corner sit inside its arc."""
    centre_reach = (
        min(float(np.hypot(*v)) for v in inner.points[1 : CORNER_STEPS + 2])
        + extent
        - PAIR_WIDTH / 2
    )
    vertex = outer.points[CORNER_STEPS // 2 + 1]
    unit = vertex / np.hypot(*vertex)
    point_reach = 0.5
    keeps = point_reach - extent + PAIR_WIDTH / 2
    """How far the point keeps from the outer track's centreline."""
    first = vertex + unit * keeps
    drawn = LineString(outer.points).distance(Point(first))
    at = first + unit * (keeps - REAL_DEPTH * inner_sagitta - drawn)
    field = Obstacles(
        [
            Obstacle("centre", Point(0.0, 0.0), centre_reach),
            Obstacle("point", Point(float(at[0]), float(at[1])), point_reach),
        ]
    )
    beside = replace(section, field=field)

    (violation,) = refused(
        lambda: optimize_chains(
            [], scenarios.PARAMS, SETTINGS, pairs=[beside], arc_chords=chords
        )
    )

    assert violation.obstacle == "point"


STRETCH_DEPTH = 0.005
"""mm a straight chain's middle starts inside its clearance of a track, its
pinned ends about as far outside it."""


def test_a_stretch_inside_is_refused_beside_a_selected_arc_elsewhere() -> None:
    # The arc's chords widen nothing the optimizer counts as inside: the
    # stretch is no run holding a pinned end.
    keep = scenarios.PAIR_INFLATION
    track = LineString(
        [
            (-1.0, -(keep + STRETCH_DEPTH)),
            (5.0, -(keep - STRETCH_DEPTH)),
            (11.0, -(keep + STRETCH_DEPTH)),
        ]
    )
    field = Obstacles([Obstacle("track t", track, keep, net="GND")])
    chain = selected("chain", "SIG", field, (0.0, 0.0), (10.0, 0.0))
    arc, points = wide_arc()
    elsewhere = selected("arc", "SIG2", FreeSpace(), *points)

    (violation,) = refused(
        lambda: optimize_chains(
            [chain, elsewhere],
            scenarios.PARAMS,
            SETTINGS,
            arc_chords=arc_chords([arc]),
        )
    )

    assert (violation.subject, violation.obstacle) == ("chain", "track t")
    assert violation.depth == pytest.approx(STRETCH_DEPTH)


SECTION_PAD = 0.25
"""mm, the radius of a round pad beside a section's track P."""
SECTION_PAD_GAP = 0.1
"""mm between such a pad and track P's copper: inside the clearance."""

SectionPad = tuple[str, str, float, float]
"""A round pad beside track P: its name, its net, its x and its gap to the
track's copper, mm."""


def section_pads(*pads: SectionPad) -> Callable[[], RunResult]:
    """A straight section P/N from x = 0 to 10 and round `pads` beside
    track P."""
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    p_y = float(straight.a.points[0][1])
    extent = (straight.gap + PAIR_WIDTH) / 2 + PAIR_WIDTH / 2
    keep = SECTION_PAD + PAIR_CLEARANCE + extent
    field = Obstacles(
        [
            Obstacle(
                name,
                Point(x, p_y + math.copysign(PAIR_WIDTH / 2 + gap + SECTION_PAD, p_y)),
                keep,
                net=net,
                extent=SECTION_PAD,
            )
            for name, net, x, gap in pads
        ]
    )
    section = replace(straight, field=field)
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])


PINNED_END_PAD: SectionPad = ("pad g", "GND", -0.2, SECTION_PAD_GAP)
"""A pad of another net inside the clearance of the section's pinned
start."""


@pytest.mark.parametrize(
    "elsewhere",
    [(), (("pad p", "P", 5.0, SECTION_PAD_GAP),)],
    ids=["alone", "beside-own-net-pad"],
)
def test_a_section_s_pinned_end_inside_a_clearance_starts(
    elsewhere: tuple[SectionPad, ...],
) -> None:
    # A pad of track P's net mid-route is a place the optimizer reports,
    # judged near it alone: the pinned end stays exempt.
    result = section_pads(PINNED_END_PAD, *elsewhere)()

    assert result.termination is not Termination.ABNORMAL


def test_a_section_s_other_net_pad_is_found_past_a_deeper_own_net_one() -> None:
    # Ranked by depth, the pad of track P's net comes first and is no
    # violation; the pad of GND beside it is. The optimizer finds one place,
    # at the deeper pad, near which the GND pad is judged.
    run = section_pads(
        ("pad p", "P", 4.6, SECTION_PAD_GAP / 2), ("pad g", "GND", 5.4, SECTION_PAD_GAP)
    )

    (violation,) = refused(run)

    assert violation.obstacle == "pad g"
    assert violation.required == pytest.approx(PAIR_CLEARANCE)


LEAD_OUT = 0.05
"""mm of the straight lead leaving the hugging arc's end."""
LEAD_PAD_GAP = 0.1
"""mm from the lead's centreline to the centre of a pad beside it."""
LEAD_DEPTH = 0.5
"""How far the lead reaches into the pad's clearance, its ends clear of it,
as a fraction of the arc's chords' sagitta: less than a chord's allowance
would hide."""


def test_a_lead_leaving_an_arc_into_a_clearance_is_refused() -> None:
    # The lead's edge cuts into the pad's clearance between two clear
    # vertices, one the arc's end: straight copper, which no chord allows.
    arc, points = round_arc()
    depth = LEAD_DEPTH * max(c.sagitta for c in arc_chords([arc]).values())
    pad = Point(-LEAD_PAD_GAP, RADIUS + LEAD_OUT / 2)
    reach = LEAD_PAD_GAP + depth
    field = Obstacles([Obstacle("pad", pad, reach, net="GND")])
    chain = selected(
        "arc", "SIG", field, (2 * RADIUS, 0.0), *points, (0.0, RADIUS + LEAD_OUT)
    )

    (violation,) = refused(
        lambda: optimize_chains(
            [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
        )
    )

    assert violation.obstacle == "pad"
    assert violation.depth == pytest.approx(depth)


DIP = 0.01
"""mm each of a chain's two dips reaches into its clearance of another."""


def test_two_chains_inside_each_other_at_two_places_are_named_once() -> None:
    # Two members are judged by their whole tracks, which gives the same
    # place for each place the optimizer finds between them.
    low = scenarios.PAIR_INFLATION - DIP
    chains = [
        selected("a", "SIG", FreeSpace(), (0.0, 0.0), (10.0, 0.0)),
        selected(
            "b",
            "OTHER",
            FreeSpace(),
            (0.0, 1.0),
            (3.0, low),
            (5.0, 1.0),
            (7.0, low),
            (10.0, 1.0),
        ),
    ]

    found = refused(lambda: optimize_chains(chains, scenarios.PARAMS, SETTINGS))

    places = [(v.subject, v.obstacle, v.at) for v in found]
    assert len(places) == len(set(places))
    assert [v.depth for v in found] == pytest.approx([DIP] * len(found))


CONVEX_EXTRA = 0.002
"""mm a pad outside the wide arc reaches into its chord's middle, its
clearance reaching the arc by that plus the chord's sagitta."""


def test_a_chord_inside_a_clearance_on_the_arc_s_convex_side_is_refused() -> None:
    # The chord is farther than the arc on the convex side: no allowance.
    found = refused(near_the_arc(deepest_sagitta() + CONVEX_EXTRA, outside=True))

    assert {v.obstacle for v in found} == {"pad"}
    assert max(v.depth for v in found) == pytest.approx(CONVEX_EXTRA, abs=NM_ROUNDING)


EDGE_INFLATION = 0.3
"""mm the board edge keeps from a chain's centreline."""
NOTCH_RADIUS = 0.5
"""mm, of a round notch in the board's outline."""
NOTCH_INTO = 0.01
"""mm the notch's clearance reaches into the wide arc's chord at its
middle."""


def test_an_arc_between_a_pad_and_the_board_edge_is_refused_by_the_edge() -> None:
    # The pad on the concave side is at the arc's clearance, its chord's
    # allowance deeper than the edge on the convex side, which no allowance
    # covers.
    arc, points = wide_arc()
    ux, uy = arc_direction(points)
    centre = (
        WIDE_RADIUS + EDGE_INFLATION + NOTCH_RADIUS - deepest_sagitta() - NOTCH_INTO
    )
    notch = Point(centre * ux, centre * uy).buffer(NOTCH_RADIUS, 256)
    board = box(-50.0, -50.0, 50.0, 50.0).difference(notch)
    pad = Obstacle(
        "pad", Point(beyond_the_arc(points, PAD_REACH, 0.0, outside=False)), PAD_REACH
    )
    chain = selected("arc", "SIG", Obstacles([pad], board, EDGE_INFLATION), *points)

    found = refused(
        lambda: optimize_chains(
            [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
        )
    )

    # Measured at the vertices the optimizer splits the chord into, which
    # sit off its middle, where the notch reaches deepest.
    assert {v.obstacle for v in found} == {"the board edge"}
    assert 0.0 < max(v.depth for v in found) <= NOTCH_INTO + NM_ROUNDING


NOTCH_DEPTH = 0.005
"""mm a straight section's outer copper is inside the board edge's
clearance along a notch."""


def test_a_section_inside_the_edge_beside_a_pad_of_its_net_is_refused() -> None:
    # The pad of track P's net is the deepest item the optimizer's measure
    # finds there, and no violation; the edge still is.
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    p_y = float(straight.a.points[0][1])
    side = math.copysign(1.0, p_y)
    extent = (straight.gap + PAIR_WIDTH) / 2 + PAIR_WIDTH / 2
    edge = side * (extent + EDGE_INFLATION - NOTCH_DEPTH)
    far = side * 5.0
    outline = Polygon(
        [
            (-5, -far),
            (15, -far),
            (15, far),
            (8, far),
            (8, edge),
            (2, edge),
            (2, far),
            (-5, far),
        ]
    )
    radius = 0.1
    y = p_y + side * (PAIR_WIDTH / 2 + OWN_NET_GAP / 2 + radius)
    own = Obstacle(
        "pad p", Point(5.0, y), radius + PAIR_CLEARANCE + extent, net="P", extent=radius
    )
    field = Obstacles([own], outline, EDGE_INFLATION + extent)
    section = replace(straight, field=field)

    found = refused(
        lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])
    )

    assert {v.obstacle for v in found} == {"the board edge"}
    assert max(v.depth for v in found) == pytest.approx(NOTCH_DEPTH)


PAD_HALF = 0.05
"""mm, half the height of a square pad a section runs over."""


def section_over_pad(y: float) -> tuple[Violation, ...]:
    """The places a straight section is refused at over a pad of GND, its
    middle at `y` across the section."""
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    extent = (straight.gap + PAIR_WIDTH) / 2 + PAIR_WIDTH / 2
    pad = Obstacle(
        "pad g",
        box(4.8, y - PAD_HALF, 5.2, y + PAD_HALF),
        PAIR_CLEARANCE + extent,
        net="GND",
    )
    section = replace(straight, field=Obstacles([pad]))
    return refused(
        lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])
    )


def track_p_y() -> float:
    """Where track P of a straight section along y = 0 runs."""
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (1.0, 0.0))
    return float(straight.a.points[0][1])


def test_a_section_over_another_net_s_pad_names_the_pad() -> None:
    # The centreline crosses the pad, between the tracks: the pad explains
    # the optimizer's depth there, not the board edge.
    found = section_over_pad(0.0)

    assert {v.obstacle for v in found} == {"pad g"}
    gap = abs(track_p_y()) - PAIR_WIDTH / 2 - PAD_HALF
    assert min(v.distance or 0.0 for v in found) == pytest.approx(gap)


def test_a_track_over_another_net_s_pad_is_as_deep_as_it_runs_into_it() -> None:
    # Inside the pad's area shapely's distance is zero: the depth runs on
    # into the pad, to its edge.
    found = section_over_pad(track_p_y())

    assert {v.obstacle for v in found} == {"pad g"}
    keep = PAIR_CLEARANCE + PAIR_WIDTH / 2
    assert max(v.depth for v in found) == pytest.approx(keep + PAD_HALF)
