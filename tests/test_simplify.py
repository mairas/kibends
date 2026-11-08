from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import BoolArray, Disc, FloatArray, FreeSpace
from scenarios import (
    PAIR_CLEARANCE,
    PAIR_WIDTH,
    PARAMS,
    ChainsScenario,
    chain,
    path,
    selected,
)
from shapely.geometry import LineString

from kibends.obstacles import ChainCentreline
from kibends.optimize import (
    ChainInput,
    OptimizerSettings,
    RunResult,
    measure,
    optimize,
    optimize_chains,
)
from kibends.simplify import (
    DEFAULT_CHORD_TOLERANCE,
    cutting_chains,
    simplify,
    simplify_chains,
)

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


COARSE = OptimizerSettings(spacing=0.3, burst_maxiter=30)
# At spacing 0.3 the drape's first burst piles vertices up; see
# test_optimize_chains.
DRAPE_SETTINGS = replace(COARSE, spacing=0.2)
# Wide enough that the drape's arch, simplified alone, cuts into the chain
# below it.
DRAPE_TOLERANCE = 0.1


def gap(a: FloatArray, b: FloatArray) -> float:
    """Smallest distance between two polylines' edges, mm."""
    return float(LineString(a).distance(LineString(b)))


def optimized(scenario: ChainsScenario, settings: OptimizerSettings) -> RunResult:
    return optimize_chains(scenario.chains, scenario.params, settings)


def simplify_run(
    scenario: ChainsScenario, run: RunResult, tolerance: float
) -> list[FloatArray]:
    return simplify_chains(scenario.chains, run, scenario.params, tolerance)


@pytest.fixture(scope="module")
def drape() -> tuple[ChainsScenario, RunResult]:
    scenario = scenarios.drape()
    return scenario, optimized(scenario, DRAPE_SETTINGS)


class TestSeveralChains:
    def test_merged_edges_keep_out_of_the_neighbours_band(
        self, drape: tuple[ChainsScenario, RunResult], log: list[str]
    ) -> None:
        scenario, run = drape
        alone = [
            simplify(c.polyline, c.fixed, given.field, DRAPE_TOLERANCE)
            for c, given in zip(run.chains, scenario.chains, strict=True)
        ]

        arch, bar = simplify_run(scenario, run, DRAPE_TOLERANCE)

        assert gap(*alone) < scenarios.PAIR_INFLATION
        assert gap(arch, bar) >= scenarios.PAIR_INFLATION
        assert len(arch) < len(run.chains[0].polyline)
        assert len(bar) < len(run.chains[1].polyline)
        assert not any(line.startswith("WARNING") for line in log)

    def test_one_chain_gives_the_single_chain_result(self) -> None:
        arc = arc_around_disc(WRAP_STANDOFF)
        fixed = free(len(arc))
        alone = ChainInput(
            ChainCentreline("arc", "N", PAIR_WIDTH, PAIR_CLEARANCE, arc),
            "F.Cu",
            fixed,
            DISC,
        )

        run = optimize_chains([alone], PARAMS, COARSE)
        (optimised,) = run.chains

        (result,) = simplify_chains([alone], run, PARAMS, WIDE_TOLERANCE)

        assert np.array_equal(
            result,
            simplify(optimised.polyline, optimised.fixed, DISC, WIDE_TOLERANCE),
        )

    def test_a_run_of_other_chains_is_refused(
        self, drape: tuple[ChainsScenario, RunResult]
    ) -> None:
        scenario, run = drape

        with pytest.raises(ValueError, match="2 optimised chains for 1"):
            simplify_chains(scenario.chains[:1], run, scenario.params)

    def test_chains_of_one_net_meeting_end_to_end_merge_up_to_the_junction(
        self, log: list[str]
    ) -> None:
        scenario = scenarios.neck_down()
        run = optimized(scenario, COARSE)

        result = simplify_run(scenario, run, DEFAULT_CHORD_TOLERANCE)

        for simplified, c in zip(result, run.chains, strict=True):
            assert np.array_equal(simplified, c.polyline[[0, -1]])
        assert not any(line.startswith("WARNING") for line in log)


def test_merged_edges_cutting_into_a_neighbour_are_found() -> None:
    # Inconsistent on purpose: the merged edge of A was never checked against B.
    a = selected("A", "A", FreeSpace(), (0.0, 0.0), (5.0, -0.5), (10.0, 0.0))
    b = selected("B", "B", FreeSpace(), (0.0, 0.3), (10.0, 0.3))
    optimised = [a.centreline.points, b.centreline.points]
    merged_a = [0, 2]
    unmerged_b = [0, 1]

    cutting = cutting_chains(
        [a, b],
        optimised,
        [a.centreline.points[merged_a], b.centreline.points],
        [merged_a, unmerged_b],
        reach=0.0,
    )

    assert cutting == [0]
