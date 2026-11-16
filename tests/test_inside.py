"""The start check: a run refuses selected copper that starts inside a
clearance as drawn, naming each place (`inside`)."""

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import pairwise

import kipy.board_types as kbt
import numpy as np
import pytest
import scenarios
from conftest import (
    NM_ROUNDING,
    PAD_HALF_SIZE,
    FakeBoard,
    build_via_obstacles,
    kipy_arc,
    kipy_pad,
    kipy_rectangle,
    kipy_via,
    layer_snapshot,
    refused,
    violations,
)
from distance_fields import Disc, FreeSpace
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType
from scenarios import PAIR_CLEARANCE, PAIR_WIDTH, selected
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box

import kibends.optimize as optimize_module
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
    ObstacleKind,
    Obstacles,
    build_obstacles,
    build_pair_obstacles,
    own_net_stops,
    own_net_vias,
    split_via_obstacles,
)
from kibends.optimize import optimize_chains
from kibends.pair_blocks import PairInput
from kibends.polylines import measure
from kibends.results import RunResult, Termination
from kibends.snapshot import Arc, LayerSnapshot, Pad, PadKind, Via
from kibends.via_moves import MovingVias
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


def test_an_arc_s_chords_within_their_sagitta_start_and_end_clear() -> None:
    chain, arc = hugging(0.0)
    placed = optimize_module._problem_of(  # pyright: ignore[reportPrivateUsage]
        [chain], [], MovingVias([chain], [], 0.0, 0.0), scenarios.PARAMS, SETTINGS
    )

    result = optimize_chains(
        [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
    )

    # A start the guard measures inside a clearance, chord by chord, which
    # it holds the run to (`_core.ProblemState.allowed`).
    assert placed.allowed[0] > 0
    assert result.termination is Termination.CONVERGED
    (got,) = result.chains
    assert violations(got.polyline, got.fixed, chain.field) == 0
    drawn = measure(chain.centreline.points, SETTINGS.spacing).length
    assert measure(got.polyline, SETTINGS.spacing).length < drawn


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
        own_vias=(),
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
    via = ViaInput(
        scenarios.through_via("v", "V", at),
        (),
        FreeSpace(),
        PAIR_CLEARANCE,
        own_vias=(),
    )
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
        own_vias=(),
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
            scenarios.through_via(name, of, (5.0, y)),
            (),
            FreeSpace(),
            PAIR_CLEARANCE,
            own_vias=(),
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
    [
        (),
        (("pad p", "P", 5.0, SECTION_PAD_GAP),),
        (("pad p", "P", 0.6, SECTION_PAD_GAP),),
    ],
    ids=["alone", "beside-own-net-pad", "beside-own-net-pad-near-the-end"],
)
def test_a_section_s_pinned_end_inside_a_clearance_is_refused(
    elsewhere: tuple[SectionPad, ...],
) -> None:
    # A pinned end is no exemption; a pad of track P's net mid-route is a
    # place the optimizer reports, judged near it alone: no violation.
    found = refused(section_pads(PINNED_END_PAD, *elsewhere))

    assert {v.obstacle for v in found} == {"pad g"}


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


UNPLATED_INSIDE = 0.05
"""mm copper starts inside the hole clearance of an unplated hole."""


def unplated_hole(net: str, x: float, y: float) -> kbt.Pad:
    """An unplated pad hole of `net`, 1 mm across, on no copper layer."""
    return kipy_pad("np", net, x, y, layers=(), pad_type=PadType.PT_NPTH, drill=1.0)


def test_a_chain_inside_an_unplated_hole_of_its_net_is_refused() -> None:
    # No copper of its net surrounds the hole, which keeps the hole
    # clearance from the chain as from any copper.
    board = FakeBoard(pads=[unplated_hole("SIG", 5.0, 0.0)])
    field = build_obstacles(layer_snapshot(board, "F.Cu"), "SIG", PAIR_WIDTH)
    (hole,) = field.items
    y = hole.inflation - UNPLATED_INSIDE
    chain = selected("chain", "SIG", field, (0.0, y), (10.0, y))

    (violation,) = refused(lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS))

    assert violation.obstacle == "hole of pad np"
    assert violation.depth == pytest.approx(UNPLATED_INSIDE)


def test_a_section_beside_an_unplated_hole_of_its_track_s_net_is_refused() -> None:
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    p_y = float(straight.a.points[0][1])
    away = math.copysign(1.0, p_y)
    extent = (straight.gap + PAIR_WIDTH) / 2 + PAIR_WIDTH / 2
    probe = FakeBoard(pads=[unplated_hole("P", 5.0, 0.0)])
    (hole,) = build_pair_obstacles(
        layer_snapshot(probe, "F.Cu"), straight.a, straight.b, straight.gap
    ).items
    keep = hole.inflation - extent + PAIR_WIDTH / 2
    """How far the hole's centre keeps from track P's centreline."""
    board = FakeBoard(
        pads=[unplated_hole("P", 5.0, p_y + away * (keep - UNPLATED_INSIDE))]
    )
    snap = layer_snapshot(board, "F.Cu")
    field = build_pair_obstacles(snap, straight.a, straight.b, straight.gap)
    section = replace(straight, field=field)

    (violation,) = refused(
        lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, pairs=[section])
    )

    assert violation.obstacle == "hole of pad np"
    assert violation.depth == pytest.approx(UNPLATED_INSIDE)


CUTOUT_INTO = 0.006
"""mm a board notch's clearance reaches into the wide arc's chord at its
middle, on the arc's convex side."""


def test_an_arc_in_a_neck_of_the_board_is_refused_by_its_convex_side() -> None:
    # A cutout inside the arc at the arc's edge clearance, its chords
    # inside by their offset and allowed; the notch outside, which the
    # same board edge holds, is judged at its own samples, no allowance.
    arc, points = wide_arc()
    ux, uy = arc_direction(points)
    centre = (
        WIDE_RADIUS + EDGE_INFLATION + NOTCH_RADIUS - deepest_sagitta() - (CUTOUT_INTO)
    )
    notch = Point(centre * ux, centre * uy).buffer(NOTCH_RADIUS, 256)
    cutout = Point(0.0, 0.0).buffer(WIDE_RADIUS - EDGE_INFLATION, 4096)
    board = box(-50.0, -50.0, 50.0, 50.0).difference(notch).difference(cutout)
    chain = selected("arc", "SIG", Obstacles([], board, EDGE_INFLATION), *points)

    found = refused(
        lambda: optimize_chains(
            [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
        )
    )

    assert {v.obstacle for v in found} == {"the board edge"}
    assert 0.0 < max(v.depth for v in found) <= CUTOUT_INTO + NM_ROUNDING


SPIKE_LENGTH = 0.08
"""mm of a chain's last edge, shorter than the spacing."""
SPIKE_AT = 0.25
"""Where along the chain's last edge a spike of the board edge points at
it, as a fraction of its length: off its middle, at a sample."""
SPIKE_DEPTH = 2e-4
"""mm the spike's clearance reaches into the chain there; its middle and
ends are clear."""


def test_an_edge_cutting_into_the_board_edge_off_its_middle_is_refused() -> None:
    # The board edge's depth varies along the edge, deepest off its middle.
    tip = (SPIKE_AT * SPIKE_LENGTH, EDGE_INFLATION - SPIKE_DEPTH)
    spike = Polygon([(tip[0] - 0.5, 1.0), tip, (tip[0] + 0.5, 1.0)])
    board = box(-5.0, -5.0, 5.0, 5.0).difference(spike)
    chain = selected(
        "chain",
        "SIG",
        Obstacles([], board, EDGE_INFLATION),
        (-1.0, -0.5),
        (0.0, 0.0),
        (SPIKE_LENGTH, 0.0),
    )

    (violation,) = refused(lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS))

    assert violation.obstacle == "the board edge"
    assert violation.depth == pytest.approx(SPIKE_DEPTH, abs=NM_ROUNDING)


PRESSED_GAP = 0.15
"""mm between a chain's copper and a section's track P: inside their
clearance."""
OWN_PAD_GAP = 0.02
"""mm between a pad of track P's net and the track's copper: deeper inside
the optimizer's measure than the chain."""


def test_a_chain_beside_a_section_past_a_pad_of_its_net_is_refused() -> None:
    # The section's run inside is deepest at the pad of its net, which is no
    # violation; the chain beside it is, from either side.
    straight = scenarios.coupled(FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    p_y = float(straight.a.points[0][1])
    side = math.copysign(1.0, p_y)
    extent = (straight.gap + PAIR_WIDTH) / 2 + PAIR_WIDTH / 2
    pad = Obstacle(
        "pad p",
        Point(5.0, p_y + side * (PAIR_WIDTH / 2 + OWN_PAD_GAP + SECTION_PAD)),
        SECTION_PAD + PAIR_CLEARANCE + extent,
        net="P",
        extent=SECTION_PAD,
    )
    section = replace(straight, field=Obstacles([pad]))
    g_y = p_y + side * (PAIR_WIDTH + PRESSED_GAP)
    chain = selected("g", "GND", FreeSpace(), (5.3, g_y), (5.3, g_y + side * 5.0))

    found = refused(
        lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS, pairs=[section])
    )

    assert {v.obstacle for v in found} == {"g", "the section of P and N"}
    assert min(v.distance or 0.0 for v in found) == pytest.approx(PRESSED_GAP)


FLAT_DEPTH = 0.05
"""mm a chain's straight flat reaches into another chain's clearance."""


def test_a_straight_flat_beside_an_arc_nearer_is_refused_by_the_arc() -> None:
    # The arc of chain B reaches nearer A than its flat does, A on its
    # convex side, where its chords are judged as they are.
    nm = round(NM_PER_MM)
    arc = Arc("arc", (0, 0), (nm, nm // 5), (2 * nm, 0), nm // 5, "F.Cu", "B")
    ends = [arc.segments()[0].start, *(s.end for s in arc.segments())]
    points = [(x / NM_PER_MM, y / NM_PER_MM) for x, y in ends]
    b = selected(
        "B",
        "B",
        FreeSpace(),
        (-10.0, -3.0),
        (-4.0, 0.0),
        *points,
        (4.0, 0.0),
        (10.0, -3.0),
    )
    h = scenarios.PAIR_INFLATION - FLAT_DEPTH
    a = selected("A", "A", FreeSpace(), (-3.0, h), (5.0, h))

    found = refused(
        lambda: optimize_chains(
            [b, a], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
        )
    )

    assert max(v.depth for v in found) > FLAT_DEPTH


WIDER_CLEARANCE = 0.3
"""mm, the netclass clearance of one of two chains, the other's
PAIR_CLEARANCE."""


def test_two_chains_keep_the_larger_of_their_clearances() -> None:
    # Their copper is between the two clearances apart.
    gap = (PAIR_CLEARANCE + WIDER_CLEARANCE) / 2
    wide = selected("wide", "W", FreeSpace(), (0.0, 0.0), (10.0, 0.0))
    wide = replace(wide, centreline=replace(wide.centreline, clearance=WIDER_CLEARANCE))
    y = PAIR_WIDTH + gap
    near = selected("near", "N", FreeSpace(), (5.0, y), (5.0, y + 5.0))

    found = refused(lambda: optimize_chains([wide, near], scenarios.PARAMS, SETTINGS))

    assert [v.required for v in found] == pytest.approx([WIDER_CLEARANCE] * len(found))


STILL_APART = 0.5
"""mm between the centres of a moving via and an unselected via of its net,
inside their hole-to-hole."""


def via_beside_its_net(apart: float) -> Callable[[], RunResult]:
    """A moving via of G and an unselected via of G `apart` above it."""
    via = scenarios.through_via("v", "G", (5.0, 1.5))
    board = FakeBoard(vias=[kipy_via("u", "G", 5.0, 1.5 + apart)])
    layers = [layer_snapshot(board, name) for name in via.span]
    given = ViaInput(
        via,
        (),
        FreeSpace(),
        PAIR_CLEARANCE,
        own_net=own_net_stops(layers, via),
        own_vias=own_net_vias(layers, via),
    )
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])


def test_a_via_inside_hole_to_hole_of_a_still_via_of_its_net_is_refused() -> None:
    required = HOLE_TO_HOLE + scenarios.VIA_DRILL

    (violation,) = refused(via_beside_its_net(STILL_APART))

    assert (violation.subject, violation.obstacle) == ("via v", "via u")
    assert violation.depth == pytest.approx(required - STILL_APART)


def test_a_via_clear_of_hole_to_hole_of_a_still_via_of_its_net_starts() -> None:
    clear = HOLE_TO_HOLE + scenarios.VIA_DRILL + 0.05

    result = via_beside_its_net(clear)()

    assert result.termination is not Termination.ABNORMAL


PAD_DRILL = 1.0
"""mm, the drill of a pad of the via's net."""
OWN_PAD_HALF = 0.8
"""mm, half the size of that pad: the via's ring overlaps it."""
DRILL_INSIDE = 0.02
"""mm the via's hole is inside hole-to-hole of the pad's drill."""


def via_over_its_pad(*, drilled: bool) -> Callable[[], RunResult]:
    """A moving via of G whose ring overlaps a pad of G; with `drilled`, its
    hole DRILL_INSIDE inside hole-to-hole of the pad's drill."""
    keep = HOLE_TO_HOLE + PAD_DRILL / 2 + scenarios.VIA_DRILL / 2
    x = keep - DRILL_INSIDE
    stop = Obstacle(
        "pad 1",
        box(-OWN_PAD_HALF, -OWN_PAD_HALF, OWN_PAD_HALF, OWN_PAD_HALF),
        scenarios.VIA_DIAMETER / 2,
    )
    hole = Obstacle(
        "hole of pad 1", Point(0.0, 0.0), keep, kind=ObstacleKind.HOLE, owner="G"
    )
    items = [stop, hole] if drilled else [stop]
    via = scenarios.through_via("v", "G", (x, 0.0))
    given = ViaInput(
        via, (), FreeSpace(), PAIR_CLEARANCE, own_net=Obstacles(items), own_vias=()
    )
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])


def test_a_via_inside_hole_to_hole_of_its_net_s_pad_drill_is_refused() -> None:
    # The pad's own stop is deeper than its drill: a stop, no rule.
    (violation,) = refused(via_over_its_pad(drilled=True))

    assert violation.obstacle == "hole of pad 1"
    assert violation.depth == pytest.approx(DRILL_INSIDE)


def test_a_via_overlapping_a_pad_of_its_net_starts() -> None:
    # The pad stops it moving onto the pad; no rule keeps it apart.
    result = via_over_its_pad(drilled=False)()

    assert result.termination is not Termination.ABNORMAL


SLOT_LENGTH = 1.7
"""mm, the long axis of a 0.6 mm wide plated slot, which the snapshot reads
as a circle of it."""
SLOT_PAD = affinity.scale(Point(0.0, 0.0).buffer(1.0), 0.5, 1.05)
"""The slot's pad: copper 1.0 x 2.1 mm round it, long axis along y."""


def slotted(net: str) -> tuple[LayerSnapshot, ...]:
    """The layers of a through via, each holding a plated slot of `net` at
    the origin."""
    pad = Pad(
        "p",
        "J1:S1",
        net,
        PadKind.PTH,
        (0.0, 0.0),
        SLOT_LENGTH,
        SLOT_PAD,
        oval_drill=True,
    )
    return tuple(
        replace(layer_snapshot(FakeBoard(), name), pads=(pad,))
        for name in ("F.Cu", "B.Cu")
    )


SLOT_SIDE = 1.0
"""mm from the slot's centre to a via's beside its narrow side: its hole
0.55 mm from the slot's edge, inside hole-to-hole of the circle the slot
reads as."""


def test_a_via_beside_a_slot_of_its_net_starts() -> None:
    # The start check leaves a hole that is not round to KiCad's DRC.
    layers = slotted("G")
    via = scenarios.through_via("v", "G", (SLOT_SIDE, 0.0))
    given = ViaInput(
        via,
        (),
        FreeSpace(),
        PAIR_CLEARANCE,
        own_net=own_net_stops(layers, via),
        own_vias=own_net_vias(layers, via),
    )

    result = optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])

    assert result.termination is not Termination.ABNORMAL


def test_a_chain_beside_a_slot_of_another_net_starts() -> None:
    # Clear of the slot's copper, inside the hole clearance of the circle
    # it reads as.
    (snap, _) = slotted("GND")
    field = build_obstacles(snap, "SIG", PAIR_WIDTH)
    x = 0.5 + PAIR_CLEARANCE + PAIR_WIDTH / 2 + 0.1
    chain = selected("chain", "SIG", field, (x, -5.0), (x, 5.0))

    result = optimize_chains([chain], scenarios.PARAMS, SETTINGS)

    assert result.termination is not Termination.ABNORMAL


SLOT_CLEAR = 0.05
"""mm a via beside another net's slot stands beyond SLOT_SIDE: its ring
clear of the slot's copper, inside the hole clearance of the circle the
slot reads as."""


def test_a_via_beside_a_slot_of_another_net_starts() -> None:
    layers = slotted("GND")
    via = scenarios.through_via("v", "S", (SLOT_SIDE + SLOT_CLEAR, 0.0))
    field, _ = split_via_obstacles(layers, via, frozenset())
    given = ViaInput(via, (), field, PAIR_CLEARANCE, own_vias=())

    result = optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])

    assert result.termination is not Termination.ABNORMAL


def test_own_net_vias_are_the_still_vias_of_its_net_on_its_layers() -> None:
    via = Via("v", "G", (0.0, 0.0), 0.6, 0.3, ("F.Cu", "In1.Cu"))
    kept = scenarios.through_via("kept", "G", (2.0, 0.0))
    other_net = scenarios.through_via("other", "H", (3.0, 0.0))
    moving = scenarios.through_via("moving", "G", (4.0, 0.0))
    below = Via("below", "G", (5.0, 0.0), 0.6, 0.3, ("B.Cu", "B.Cu"))
    layers = [
        replace(
            layer_snapshot(FakeBoard(), "F.Cu"), vias=(via, kept, other_net, moving)
        ),
        replace(layer_snapshot(FakeBoard(), "B.Cu"), vias=(below,)),
    ]

    found = own_net_vias(layers, via, moving=frozenset({"moving"}))

    assert [v.id for v in found] == ["kept"]


def test_a_flat_inside_beside_an_arc_is_refused_on_both_sides() -> None:
    # A's one drawn edge is nearest B at the arc, on whose convex side A
    # lies, where the chords are judged as they are, deeper than the flat;
    # B's run inside, its pinned end included, is refused too.
    nm = round(NM_PER_MM)
    arc = Arc("arc", (0, 0), (nm, nm // 5), (2 * nm, 0), nm // 5, "F.Cu", "B")
    ends = [arc.segments()[0].start, *(s.end for s in arc.segments())]
    points = [(x / NM_PER_MM, y / NM_PER_MM) for x, y in ends]
    b = selected("B", "B", FreeSpace(), (-4.0, 0.0), *points, (4.0, 0.0), (10.0, -3.0))
    h = scenarios.PAIR_INFLATION - FLAT_DEPTH
    a = selected("A", "A", FreeSpace(), (-5.0, h), (5.0, h))

    found = refused(
        lambda: optimize_chains(
            [b, a], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
        )
    )

    assert {v.subject for v in found} == {"A", "B"}
    assert max(v.depth for v in found if v.subject == "A") > FLAT_DEPTH


def u_section(depth: float) -> PairInput:
    """A section from a pinned start at the origin round a U back along
    y = -depth to x = -7, where it is pinned again."""
    return scenarios.coupled(
        FreeSpace(), (0.0, 0.0), (3.0, 0.0), (3.0, -depth), (-7.0, -depth)
    )


START_ONLY = replace(SETTINGS, coarse_spacings=(), max_iterations=1)
"""Settings for a test of whether a run starts, which need not optimize."""
BESIDE_START = 0.3
"""mm between a chain's pinned start and another chain's pinned end: inside
their clearance."""
ARC_DIP = 0.35
"""mm an arc of the chain dips to from the other chain: inside their
clearance."""


def test_a_chain_inside_where_it_is_pinned_is_refused_along_its_arc() -> None:
    # X's start and Y's end, both pinned, lie inside each other's clearance;
    # further on X's arc dips inside Y's, on whose convex side Y lies: the
    # chords are judged as they are, along that run.
    nm = round(NM_PER_MM)
    y = -BESIDE_START
    bottom = y + ARC_DIP
    arc = Arc(
        "arc",
        (4 * nm, 2 * nm),
        (5 * nm, round(bottom * nm)),
        (6 * nm, 2 * nm),
        nm // 20,
        "F.Cu",
        "X",
    )
    ends = [arc.segments()[0].start, *(s.end for s in arc.segments())]
    points = [(px / NM_PER_MM, py / NM_PER_MM) for px, py in ends]
    x = selected("x", "X", FreeSpace(), (0.0, 0.0), (0.0, 2.0), *points, (10.0, 2.0))
    other = selected("y", "Y", FreeSpace(), (0.0, y), (8.0, y))

    found = refused(
        lambda: optimize_chains(
            [x, other], scenarios.PARAMS, START_ONLY, arc_chords=arc_chords([arc])
        )
    )

    assert {v.subject for v in found} == {"x", "y"}


def test_a_via_beside_a_section_s_pinned_start_is_refused() -> None:
    # A moving via inside the clearance of the section's pinned start, and
    # past its run just beyond its tracks' clearance.
    at = VIA_BEYOND - 0.125
    section = u_section(at + VIA_BEYOND + NEAR)
    via = ViaInput(
        scenarios.through_via("v", "V", (0.0, -at)),
        (),
        FreeSpace(),
        PAIR_CLEARANCE,
        own_vias=(),
    )

    found = refused(
        lambda: optimize_chains(
            [], scenarios.PARAMS, START_ONLY, vias=[via], pairs=[section]
        )
    )

    assert {v.obstacle for v in found} == {"the section of P and N"}


BETWEEN_GAP = 0.15
"""mm between a chain's copper and each of two chains on either side of it:
inside their clearance."""


def test_a_run_inside_two_chains_names_both() -> None:
    y = PAIR_WIDTH + BETWEEN_GAP
    chains = [
        selected("x", "X", FreeSpace(), (0.0, 0.0), (10.0, 0.0)),
        selected("above", "A", FreeSpace(), (4.0, y), (6.0, y)),
        selected("below", "B", FreeSpace(), (4.0, -y), (6.0, -y)),
    ]

    found = refused(lambda: optimize_chains(chains, scenarios.PARAMS, SETTINGS))

    assert {v.obstacle for v in found if v.subject == "x"} == {"above", "below"}


STATIC_RADIUS = 30.0
"""mm, of an unselected quarter arc of another net about the origin."""
STATIC_WIDTH = 0.25
"""mm, its width."""


def static_arc_layer(
    *pads: kbt.Pad, radius: float = STATIC_RADIUS, width: float = STATIC_WIDTH
) -> LayerSnapshot:
    """F.Cu with an unselected quarter arc of OTHER of `radius` about the
    origin, and `pads`."""
    d = radius / math.sqrt(2)
    arc = kipy_arc("o", "OTHER", (radius, 0.0), (d, d), (0.0, radius), width)
    return layer_snapshot(FakeBoard(tracks=[arc], pads=list(pads)), "F.Cu")


@pytest.mark.parametrize("radius", [3.0, 10.0, 30.0, 100.0])
def test_a_via_at_clearance_inside_an_unselected_arc_starts(radius: float) -> None:
    # The arc is held as its chords, which sit nearer the via than the arc
    # by up to their sagitta: on its concave side a chord is allowed the
    # arc's offset from it.
    snap = static_arc_layer(radius=radius)
    angle = chord_middle(snap)
    keep = PAIR_CLEARANCE + scenarios.VIA_DIAMETER / 2 + STATIC_WIDTH / 2
    at = (
        (radius - keep) * math.cos(angle),
        (radius - keep) * math.sin(angle),
    )
    via = scenarios.through_via("v", "V", at)
    given = ViaInput(
        via, (), build_via_obstacles([snap], via), PAIR_CLEARANCE, own_vias=()
    )

    result = optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])

    assert result.termination is not Termination.ABNORMAL


ROUND_INSIDE = 0.02
"""mm copper starts inside the clearance of a round pad."""


def chord_middle(snap: LayerSnapshot) -> float:
    """The angle of the middle of the unselected arc's middle chord."""
    coords = list(snap.tracks[0].centreline.coords)
    k = len(coords) // 2
    (ax, ay), (bx, by) = coords[k], coords[k + 1]
    return (math.atan2(ay, ax) + math.atan2(by, bx)) / 2


def test_a_chain_beside_an_unselected_arc_inside_a_pad_s_clearance_is_refused() -> None:
    # The chain's free tip reaches the arc's clearance at a chord's middle,
    # where the chords are nearer; the arc is skipped, the pad beside the
    # tip is not.
    angle = chord_middle(static_arc_layer())
    ux, uy = math.cos(angle), math.sin(angle)
    keep = scenarios.PAIR_INFLATION + (STATIC_WIDTH - PAIR_WIDTH) / 2
    tx, ty = (STATIC_RADIUS - keep) * ux, (STATIC_RADIUS - keep) * uy
    # The square pad's support toward the tip, which meets it slanted.
    support = PAD_HALF_SIZE * (abs(ux) + abs(uy))
    reach = PAIR_WIDTH / 2 + PAIR_CLEARANCE + support - ROUND_INSIDE
    snap = static_arc_layer(kipy_pad("q", "GND", tx - reach * ux, ty - reach * uy))
    field = build_obstacles(snap, "SIG", PAIR_WIDTH)
    side = 2.0
    chain = selected(
        "chain",
        "SIG",
        field,
        (tx - side * (ux - uy), ty - side * (uy + ux)),
        (tx, ty),
        (tx - side * (ux + uy), ty - side * (uy - ux)),
    )

    found = refused(lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS))

    assert {v.obstacle for v in found} == {"pad q"}


def tip_by_the_arc(inside: float) -> Callable[[], RunResult]:
    """A straight chain's free tip `inside` the clearance of the unselected
    arc, on its concave side at a chord's middle."""
    snap = static_arc_layer()
    radius = STATIC_RADIUS
    angle = chord_middle(snap)
    ux, uy = math.cos(angle), math.sin(angle)
    keep = scenarios.PAIR_INFLATION + (STATIC_WIDTH - PAIR_WIDTH) / 2
    tx = (radius - keep + inside) * ux
    ty = (radius - keep + inside) * uy
    field = build_obstacles(snap, "SIG", PAIR_WIDTH)
    side = 2.0
    chain = selected(
        "chain",
        "SIG",
        field,
        (tx - side * (ux - uy), ty - side * (uy + ux)),
        (tx, ty),
        (tx - side * (ux + uy), ty - side * (uy - ux)),
    )
    return lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS)


def test_a_chain_at_clearance_inside_an_unselected_arc_starts() -> None:
    # Its chords sit nearer the tip than the arc.
    result = tip_by_the_arc(0.0)()

    assert result.termination is not Termination.ABNORMAL


def via_beside_slot_and(
    *, pads: Sequence[kbt.Pad] = (), shapes: Sequence[kbt.BoardRectangle] = ()
) -> Callable[[], RunResult]:
    """A via of S beside a slot of GND, inside the circle the slot reads as,
    with `pads` and `shapes` on its layers."""
    extra = layer_snapshot(FakeBoard(pads=list(pads), shapes=list(shapes)), "F.Cu")
    layers = [
        replace(extra, layer=slot.layer, pads=(*slot.pads, *extra.pads))
        for slot in slotted("GND")
    ]
    via = scenarios.through_via("v", "S", (SLOT_SIDE + SLOT_CLEAR, 0.0))
    field, _ = split_via_obstacles(layers, via, frozenset())
    given = ViaInput(via, (), field, PAIR_CLEARANCE, own_vias=())
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])


VIA_BESIDE_SLOT = SLOT_SIDE + SLOT_CLEAR
"""Where the via beside the slot stands, mm along x."""


def test_a_via_beside_a_slot_and_inside_a_pad_s_clearance_is_refused() -> None:
    reach = scenarios.VIA_DIAMETER / 2 + PAIR_CLEARANCE + PAD_HALF_SIZE
    pad = kipy_pad("q", "X", VIA_BESIDE_SLOT + reach - ROUND_INSIDE, 0.0)

    (violation,) = refused(via_beside_slot_and(pads=[pad]))

    assert violation.obstacle == "pad q"
    assert violation.depth == pytest.approx(ROUND_INSIDE)


def board_edge_beyond_the_via(gap: float) -> kbt.BoardRectangle:
    """An outline whose edge stands `gap` beyond the edge clearance of the
    via beside the slot."""
    reach = scenarios.VIA_DIAMETER / 2 + EDGE_CLEARANCE
    return kipy_rectangle(
        (-10.0, -10.0), (VIA_BESIDE_SLOT + reach + gap, 10.0), layer=BL.BL_Edge_Cuts
    )


def test_a_via_beside_a_slot_and_inside_the_edge_s_clearance_is_refused() -> None:
    (violation,) = refused(
        via_beside_slot_and(shapes=[board_edge_beyond_the_via(-ROUND_INSIDE)])
    )

    assert violation.obstacle == "the board edge"
    assert violation.depth == pytest.approx(ROUND_INSIDE)


def test_a_via_beside_a_slot_and_clear_of_the_edge_starts() -> None:
    result = via_beside_slot_and(shapes=[board_edge_beyond_the_via(ROUND_INSIDE)])()

    assert result.termination is not Termination.ABNORMAL


OUTER_RADIUS = 100.0
"""mm, of a selected quarter arc beside an unselected concentric one of
another net outside it."""


def beside_an_outer_arc(nearer: float) -> Callable[[], RunResult]:
    """The selected arc of OUTER_RADIUS, and the unselected one outside it at
    their clearance but `nearer`: their chords 18 and 19."""
    r = round(OUTER_RADIUS * NM_PER_MM)
    d = round(OUTER_RADIUS * NM_PER_MM / math.sqrt(2))
    arc = Arc("a", (r, 0), (d, d), (0, r), round(PAIR_WIDTH * NM_PER_MM), "F.Cu", "SIG")
    ends = [arc.segments()[0].start, *(s.end for s in arc.segments())]
    points = [(x / NM_PER_MM, y / NM_PER_MM) for x, y in ends]
    outer = OUTER_RADIUS + PAIR_CLEARANCE + PAIR_WIDTH / 2 + STATIC_WIDTH / 2 - nearer
    snap = static_arc_layer(radius=outer)
    assert (len(points) - 1, len(snap.tracks[0].centreline.coords) - 1) == (18, 19)
    chain = selected("arc", "SIG", build_obstacles(snap, "SIG", PAIR_WIDTH), *points)
    return lambda: optimize_chains(
        [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
    )


def test_an_arc_at_clearance_inside_an_unselected_concentric_arc_starts() -> None:
    # Each arc's chords sit inside it: the selected one's farther from the
    # outer arc, the unselected one's nearer, allowed on its concave side.
    result = beside_an_outer_arc(0.0)()

    assert result.termination is not Termination.ABNORMAL


def test_an_arc_just_inside_an_unselected_concentric_arc_s_clearance_is_refused() -> (
    None
):
    found = refused(beside_an_outer_arc(2 * START_TOLERANCE))

    assert {v.obstacle for v in found} == {"track o"}


TWO_ARCS_RADIUS = 30.0
"""mm, of the inner of two selected concentric quarter arcs."""


def test_two_selected_concentric_arcs_at_clearance_start() -> None:
    # Their chord counts differ: each chord is allowed toward the other on
    # its own concave side, the outer arc's toward the inner.
    arcs: list[Arc] = []
    chains: list[ChainInput] = []
    for name, net, radius in (
        ("inner", "A", TWO_ARCS_RADIUS),
        ("outer", "B", TWO_ARCS_RADIUS + PAIR_CLEARANCE + PAIR_WIDTH + 1e-6),
    ):
        r = round(radius * NM_PER_MM)
        d = round(radius * NM_PER_MM / math.sqrt(2))
        arc = Arc(
            name, (r, 0), (d, d), (0, r), round(PAIR_WIDTH * NM_PER_MM), "F.Cu", net
        )
        ends = [arc.segments()[0].start, *(s.end for s in arc.segments())]
        points = [(x / NM_PER_MM, y / NM_PER_MM) for x, y in ends]
        arcs.append(arc)
        chains.append(selected(name, net, FreeSpace(), *points))
    inner, outer = (len(c.centreline.points) for c in chains)
    assert inner != outer

    result = optimize_chains(
        chains, scenarios.PARAMS, SETTINGS, arc_chords=arc_chords(arcs)
    )

    assert result.termination is not Termination.ABNORMAL


BEYOND_DEPTH = 0.005
"""mm a notch of the board edge beyond the arc's last chord reaches into
the chord's end."""
NOTCH_SMALL = 0.1
"""mm, the radius of that notch."""


def test_the_board_edge_beyond_a_chord_s_end_is_judged_without_its_allowance() -> None:
    # The notch lies past the end of the arc's last chord, on its convex
    # side, beside the chord's line; the chain turns back toward the arc's
    # centre there, away from it.
    arc, points = round_arc()
    end = points[-1]
    reach = EDGE_INFLATION - BEYOND_DEPTH + NOTCH_SMALL
    beyond = NOTCH_SMALL + 0.01
    across = math.sqrt(reach * reach - beyond * beyond)
    notch = Point(end[0] - beyond, end[1] + across).buffer(NOTCH_SMALL, 256)
    board = box(-5.0, -5.0, 5.0, 5.0).difference(notch)
    chain = selected(
        "arc",
        "SIG",
        Obstacles([], board, EDGE_INFLATION),
        (RADIUS + LEAD, 0.0),
        *points,
        (0.0, 0.0),
    )

    found = refused(
        lambda: optimize_chains(
            [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
        )
    )

    assert {v.obstacle for v in found} == {"the board edge"}


DEEP_INSIDE = 0.02
"""mm a via or a chain stands inside an unselected arc's clearance."""
OFF_MIDDLE = math.radians(46.25)
"""The angle a via beside the unselected arc stands at: off its chords'
middles and ends."""


def via_by_the_arc(radius: float, at: float) -> Callable[[], RunResult]:
    """A moving via of V at `at` from the centre of an unselected arc of
    `radius`, at OFF_MIDDLE."""
    snap = static_arc_layer(radius=radius)
    via = scenarios.through_via(
        "v", "V", (at * math.cos(OFF_MIDDLE), at * math.sin(OFF_MIDDLE))
    )
    given = ViaInput(
        via, (), build_via_obstacles([snap], via), PAIR_CLEARANCE, own_vias=()
    )
    return lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])


VIA_TO_ARC = PAIR_CLEARANCE + scenarios.VIA_DIAMETER / 2 + STATIC_WIDTH / 2
"""mm from a via's centre to the unselected arc's centreline at their
clearance."""


@pytest.mark.parametrize(
    ("radius", "outside"),
    [(3.0, False), (30.0, False), (3.0, True)],
    ids=["concave-3", "concave-30", "convex-3"],
)
def test_a_via_inside_an_unselected_arc_s_clearance_is_refused(
    radius: float, outside: bool
) -> None:
    # On the convex side of a 30 mm arc its chord stands farther than this
    # from the arc there, so that the via is clear of it and starts.
    at = (
        radius + VIA_TO_ARC - DEEP_INSIDE
        if outside
        else radius - VIA_TO_ARC + DEEP_INSIDE
    )

    (violation,) = refused(via_by_the_arc(radius, at))

    assert violation.obstacle == "track o"


CHORD_INSIDE = 0.01
"""mm copper on an unselected arc's convex side stands inside its chord's
clearance at the chord's middle."""


def outside_a_chord(radius: float, keep: float) -> tuple[Point2, Point2]:
    """Where what keeps `keep` from the unselected arc's chords stands on
    their convex side, CHORD_INSIDE inside it at a middle chord's middle, and
    the unit direction out from the arc's centre there."""
    coords = list(static_arc_layer(radius=radius).tracks[0].centreline.coords)
    k = len(coords) // 2
    (ax, ay), (bx, by) = coords[k], coords[k + 1]
    mx, my = (ax + bx) / 2, (ay + by) / 2
    n = math.hypot(mx, my)
    out = (mx / n, my / n)
    reach = n + keep - CHORD_INSIDE
    return (reach * out[0], reach * out[1]), out


@pytest.mark.parametrize("radius", [30.0, 100.0])
def test_a_via_outside_an_unselected_arc_inside_its_chord_s_clearance_is_refused(
    radius: float,
) -> None:
    # On the convex side the chord stands farther than the arc: no allowance.
    at, _ = outside_a_chord(radius, VIA_TO_ARC)
    snap = static_arc_layer(radius=radius)
    via = scenarios.through_via("v", "V", at)
    given = ViaInput(
        via, (), build_via_obstacles([snap], via), PAIR_CLEARANCE, own_vias=()
    )

    (violation,) = refused(
        lambda: optimize_chains([], scenarios.PARAMS, SETTINGS, vias=[given])
    )

    assert violation.obstacle == "track o"
    assert violation.depth == pytest.approx(CHORD_INSIDE, abs=NM_ROUNDING)


@pytest.mark.parametrize("radius", [30.0, 100.0])
def test_a_chain_outside_an_unselected_arc_inside_its_chord_s_clearance_is_refused(
    radius: float,
) -> None:
    keep = scenarios.PAIR_INFLATION + (STATIC_WIDTH - PAIR_WIDTH) / 2
    (tx, ty), (ux, uy) = outside_a_chord(radius, keep)
    snap = static_arc_layer(radius=radius)
    side = 2.0
    chain = selected(
        "chain",
        "SIG",
        build_obstacles(snap, "SIG", PAIR_WIDTH),
        (tx + side * (ux - uy), ty + side * (uy + ux)),
        (tx, ty),
        (tx + side * (ux + uy), ty + side * (uy - ux)),
    )

    found = refused(lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS))

    assert {v.obstacle for v in found} == {"track o"}


def test_a_chain_inside_an_unselected_arc_beyond_its_allowance_is_refused() -> None:
    found = refused(tip_by_the_arc(DEEP_INSIDE))

    assert {v.obstacle for v in found} == {"track o"}
