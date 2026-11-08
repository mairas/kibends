import numpy as np
import pytest
from distance_fields import FreeSpace

from kibends.chain import Chain, Point2
from kibends.coords import BOARD_FRAME, UserFrame
from kibends.energy import FloatArray, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.pair_layout import ChainPair, find_pairs, lay_out
from kibends.simplify import DEFAULT_CHORD_TOLERANCE
from kibends.vias import ChainEnd

WIDTH = 0.2
GAP = 0.15
HALF_PITCH = (GAP + WIDTH) / 2
FAN = 1.0 + HALF_PITCH
CHORD = DEFAULT_CHORD_TOLERANCE


def chain(
    net: str, *points: Point2, width: float = WIDTH, layer: str = "F.Cu"
) -> Chain:
    return Chain(
        points=points,
        pinned=(True,) + (False,) * (len(points) - 2) + (True,),
        width=width,
        layer=layer,
        net=net,
        segment_ids=(),
    )


def fanned(
    net: str,
    side: float,
    *,
    width: float = WIDTH,
    layer: str = "F.Cu",
    at: float = 0.0,
) -> Chain:
    """Coupled from x = 1 to x = 9 along y = `at`, fanned out at 45° at both
    ends."""
    return chain(
        net,
        (0.0, at + side * FAN),
        (1.0, at + side * HALF_PITCH),
        (9.0, at + side * HALF_PITCH),
        (10.0, at + side * FAN),
        width=width,
        layer=layer,
    )


def far(net: str) -> Chain:
    return chain(net, (0.0, 5.0), (10.0, 5.0))


def test_pairs_two_coupled_chains_of_a_pair_of_nets() -> None:
    chains = [far("GND"), fanned("USB_D+", -1), fanned("USB_D-", 1)]

    found = find_pairs(chains, CHORD, frame=BOARD_FRAME)

    assert [(p.a, p.b) for p in found] == [(1, 2)]
    assert found[0].pair.gap == pytest.approx(GAP)


def test_chains_of_a_pair_on_two_layers_do_not_pair(log: list[str]) -> None:
    chains = [fanned("USB_D+", -1), fanned("USB_D-", 1, layer="B.Cu")]

    assert find_pairs(chains, CHORD, frame=BOARD_FRAME) == []
    assert not any(line.startswith("WARNING") for line in log)


def test_nets_that_pair_but_do_not_run_coupled_warn(log: list[str]) -> None:
    chains = [fanned("USB_D+", -1), far("USB_D-")]

    assert find_pairs(chains, CHORD, frame=BOARD_FRAME) == []
    (warning,) = [line for line in log if line.startswith("WARNING")]
    assert "USB_D+" in warning and "USB_D-" in warning
    assert "do not run coupled" in warning


def test_pair_of_two_widths_warns_and_stays_two_chains(log: list[str]) -> None:
    chains = [fanned("USB_D+", -1), fanned("USB_D-", 1, width=0.25)]

    assert find_pairs(chains, CHORD, frame=BOARD_FRAME) == []
    (warning,) = [line for line in log if line.startswith("WARNING")]
    assert "two track widths" in warning and "two chains" in warning


def beyond(net: str) -> Chain:
    """Coupled with `fanned(..., 1)` on the side away from `fanned(..., -1)`."""
    y = 3 * HALF_PITCH
    return chain(net, (0.0, y + 1.0), (1.0, y), (9.0, y), (10.0, y + 1.0))


def test_a_chain_pairs_only_once_and_its_other_partner_warns(log: list[str]) -> None:
    chains = [fanned("USB_D+", -1), fanned("USB_D-", 1), beyond("USB_D+")]

    found = find_pairs(chains, CHORD, frame=UserFrame((1.0, 2.0)))

    assert [(p.a, p.b) for p in found] == [(0, 1)]
    (warning,) = [line for line in log if line.startswith("WARNING")]
    assert f"USB_D+ on F.Cu from (-1.000, {3 * HALF_PITCH - 1.0:.3f})" in warning
    assert "pairs with another chain of USB_D+" in warning


def test_coupled_chains_of_nets_that_do_not_pair_do_not_warn(
    log: list[str],
) -> None:
    chains = [fanned("DATA_A", -1), fanned("DATA_B", 1)]

    assert find_pairs(chains, CHORD, frame=BOARD_FRAME) == []
    assert not any(line.startswith("WARNING") for line in log)


def test_a_chain_far_from_a_paired_partner_does_not_warn(log: list[str]) -> None:
    chains = [fanned("USB_D+", -1), fanned("USB_D-", 1), far("USB_D+")]

    assert [(p.a, p.b) for p in find_pairs(chains, CHORD, frame=BOARD_FRAME)] == [
        (0, 1)
    ]
    assert not any(line.startswith("WARNING") for line in log)


def test_two_pairs_of_the_same_nets_side_by_side_do_not_warn(log: list[str]) -> None:
    # The inner tracks of the two pairs, of complementary nets, run coupled.
    chains = [
        fanned("USB_D+", -1),
        fanned("USB_D-", 1),
        fanned("USB_D+", -1, at=4 * HALF_PITCH),
        fanned("USB_D-", 1, at=4 * HALF_PITCH),
    ]

    assert [(p.a, p.b) for p in find_pairs(chains, CHORD, frame=BOARD_FRAME)] == [
        (0, 1),
        (2, 3),
    ]
    assert not any(line.startswith("WARNING") for line in log)


def test_a_chain_pairing_later_leaves_no_warning_from_before(log: list[str]) -> None:
    chains = [far("USB_D+"), fanned("USB_D-", 1), fanned("USB_D+", -1)]

    assert [(p.a, p.b) for p in find_pairs(chains, CHORD, frame=BOARD_FRAME)] == [
        (1, 2)
    ]
    assert not any(line.startswith("WARNING") for line in log)


def given(c: Chain, field: ObstacleField | None = None) -> ChainInput:
    points = np.array(c.points, dtype=float)
    return ChainInput(
        ChainCentreline(c.net, c.net, c.width, 0.2, points),
        c.layer,
        np.array(c.pinned),
        FreeSpace() if field is None else field,
    )


def reversed_chain(c: Chain) -> Chain:
    return chain(c.net, *c.points[::-1], width=c.width, layer=c.layer)


def free(
    _layer: str, _a: ChainCentreline, _b: ChainCentreline, _gap: float
) -> ObstacleField:
    return FreeSpace()


def laid_out(chains: list[Chain]) -> tuple[list[ChainPair], list[ChainInput]]:
    return find_pairs(chains, CHORD, frame=BOARD_FRAME), [given(c) for c in chains]


@pytest.mark.parametrize("reverse", [False, True])
def test_layout_joins_unchanged_parts_back_into_the_pair_tracks(
    reverse: bool,
) -> None:
    b = fanned("USB_D-", 1)
    chains = [far("GND"), fanned("USB_D+", -1), reversed_chain(b) if reverse else b]
    pairs, whole = laid_out(chains)
    section_fields: list[tuple[str, str, str, float]] = []

    def field(
        layer: str, a: ChainCentreline, b: ChainCentreline, gap: float
    ) -> ObstacleField:
        section_fields.append((layer, a.net, b.net, gap))
        return FreeSpace()

    layout = lay_out(whole, pairs, field)

    assert layout.inputs[0] is whole[0]
    assert layout.slots == [0, None, None]
    assert len(layout.inputs) == 1 + 4
    assert section_fields == [("F.Cu", "USB_D+", "USB_D-", pytest.approx(GAP))]
    joined = layout.whole_polylines(
        [i.centreline.points for i in layout.inputs],
        [(s.a.points, s.b.points) for s in layout.sections],
    )
    (pair,) = pairs
    np.testing.assert_array_equal(joined[0], whole[0].centreline.points)
    np.testing.assert_array_equal(joined[1], pair.pair.a.points)
    np.testing.assert_array_equal(joined[2], pair.pair.b.points)


WIDER_END = 0.012
"""How much wider than the rest the gap at one end of a coupled stretch is,
mm, within what still pairs."""


def test_a_section_s_field_is_as_wide_as_its_widest_end() -> None:
    wide = HALF_PITCH + WIDER_END
    chains = [
        fanned("USB_D+", -1),
        chain(
            "USB_D-",
            (0.0, FAN),
            (1.0, HALF_PITCH),
            (8.0, HALF_PITCH),
            (8.1, wide),
            (9.0, wide),
            (10.0, FAN),
        ),
    ]
    pairs, whole = laid_out(chains)
    gaps: list[float] = []

    def field(
        _layer: str, _a: ChainCentreline, _b: ChainCentreline, gap: float
    ) -> ObstacleField:
        gaps.append(gap)
        return FreeSpace()

    lay_out(whole, pairs, field)

    assert gaps == [pytest.approx(GAP + WIDER_END)]


SECOND_PAIR = 5.0
"""y of the second pair's centreline, mm."""


def test_layout_of_two_pairs_gives_each_its_own_sections() -> None:
    chains = [
        fanned("USB_D+", -1),
        fanned("ETH_P", -1, at=SECOND_PAIR),
        fanned("USB_D-", 1),
        fanned("ETH_N", 1, at=SECOND_PAIR),
    ]
    pairs, whole = laid_out(chains)

    layout = lay_out(whole, pairs, free)

    assert [(p.a, p.b) for p in pairs] == [(0, 2), (1, 3)]
    assert layout.pair_sections == [range(0, 1), range(1, 2)]
    assert [(s.a.net, s.b.net) for s in layout.sections] == [
        ("USB_D+", "USB_D-"),
        ("ETH_P", "ETH_N"),
    ]
    joined = layout.whole_polylines(
        [i.centreline.points for i in layout.inputs],
        [(s.a.points, s.b.points) for s in layout.sections],
    )
    for p in pairs:
        np.testing.assert_array_equal(joined[p.a], p.pair.a.points)
        np.testing.assert_array_equal(joined[p.b], p.pair.b.points)


def test_layout_moves_via_ends_to_the_chains_new_places() -> None:
    chains = [fanned("USB_D+", -1), fanned("USB_D-", 1), far("GND")]
    pairs, whole = laid_out(chains)

    layout = lay_out(whole, pairs, free)

    assert layout.ends((ChainEnd(2, True),)) == (ChainEnd(0, True),)


def test_layout_without_pairs_is_the_chains() -> None:
    chains = [far("GND"), far("SIG")]
    pairs, whole = laid_out(chains)

    layout = lay_out(whole, pairs, free)

    assert layout.inputs == whole
    assert layout.sections == []
    polylines: list[FloatArray] = [w.centreline.points for w in whole]
    assert all(
        j is p
        for j, p in zip(layout.whole_polylines(polylines, []), polylines, strict=True)
    )
