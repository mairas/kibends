"""The start check: a run refuses selected copper that starts inside a
clearance as drawn, naming each place (`inside`)."""

import math
from collections.abc import Callable
from dataclasses import replace

import kipy.board_types as kbt
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
from distance_fields import FreeSpace
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType
from scenarios import PAIR_CLEARANCE, PAIR_WIDTH, selected
from shapely.geometry import Point, Polygon, box

from kibends.bursts import DEFAULT_SETTINGS, INSIDE_TOLERANCE
from kibends.chain import NM_PER_MM, Point2
from kibends.coords import UserFrame
from kibends.inside import (
    INSIDE_MESSAGE,
    InsideClearanceError,
    Violation,
    arc_chords,
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


EDGE_INFLATION = 0.3
"""mm the board edge keeps from a chain's centreline."""


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


ALONG_THE_ARC = 0.01
"""mm a pad outside the round arc reaches into it at its middle vertex:
truly inside, its leads clear."""
ALONG_REACH = 0.5
"""mm, how far the pad's clearance reaches from its centre."""


def test_a_place_along_an_arc_s_chords_is_skipped() -> None:
    # The start check leaves the drawn arc to KiCad's DRC.
    arc, points = round_arc()
    at = (RADIUS + ALONG_REACH - ALONG_THE_ARC) / math.sqrt(2)
    field = Obstacles([Obstacle("pad", Point(at, at), ALONG_REACH)])
    chain = selected(
        "arc", "SIG", field, (RADIUS + LEAD, 0.0), *points, (0.0, RADIUS + LEAD)
    )

    assert refused(lambda: optimize_chains([chain], scenarios.PARAMS, SETTINGS))
    result = optimize_chains(
        [chain], scenarios.PARAMS, SETTINGS, arc_chords=arc_chords([arc])
    )
    assert result.termination is not Termination.ABNORMAL


PRESSED_GAP = 0.15
"""mm between a chain's copper and a section's track P: inside their
clearance."""
OWN_PAD_GAP = 0.02
"""mm between a pad of track P's net and the track's copper: deeper inside
the optimizer's measure than the chain."""


def test_a_chain_beside_a_section_past_a_pad_of_its_net_is_refused() -> None:
    # The section's run inside is deepest at the pad of its net, which is no
    # violation; the chain in the same run is.
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

    assert {v.obstacle for v in found} == {"g"}
    assert min(v.distance or 0.0 for v in found) == pytest.approx(PRESSED_GAP)


FLAT_DEPTH = 0.05
"""mm a chain's straight flat reaches into another chain's clearance."""


def test_a_straight_flat_beside_a_skipped_arc_is_refused() -> None:
    # The arc of chain B reaches nearer A than its flat does; places along
    # its chords are skipped, the flat is not.
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

    assert max(v.depth for v in found) == pytest.approx(FLAT_DEPTH)


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
