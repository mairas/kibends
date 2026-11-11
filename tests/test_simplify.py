from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import clearances
from distance_fields import BoolArray, Disc, FloatArray, FreeSpace
from reference_fields import chain_obstacles
from scenarios import (
    DP_HALF_EXTENT,
    PAIR_CLEARANCE,
    PAIR_WIDTH,
    PARAMS,
    ChainsScenario,
    chain,
    path,
)
from shapely.geometry import LineString

from kibends.bursts import INSIDE_TOLERANCE, OptimizerSettings
from kibends.contacts import no_contacts
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline, track_inflation
from kibends.optimize import optimize, optimize_chains
from kibends.pair_blocks import PairBlock, pair_block
from kibends.polylines import measure
from kibends.results import ChainResult, RunResult, Termination
from kibends.simplify import simplify, simplify_chains

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

        assert clearances(tolerance_only, DISC).min() < 0.0
        assert len(tolerance_only) < len(result) < len(arc)
        assert DISC.segments_clear(result[:-1], result[1:]).all()
        assert clearances(result, DISC).min() >= 0.0

    def test_unmergeable_wrap_is_left_alone(self) -> None:
        arc = arc_around_disc(standoff=0.001)

        result = simplify(arc, free(len(arc)), DISC, WIDE_TOLERANCE)

        assert np.array_equal(result, arc)


def test_optimized_straight_line_simplifies_to_one_segment() -> None:
    polyline, fixed = chain((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    # Leaves a bow of about 15 µm for simplify to flatten, as the defaults do.
    settings = OptimizerSettings(spacing=0.2, settled_change=0.002, burst_maxiter=100)
    optimized = optimize(polyline, fixed, FreeSpace(), PARAMS, settings)

    result = simplify(optimized.polyline, optimized.fixed, FreeSpace())

    assert np.array_equal(result, polyline[[0, -1]])


COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)
# Above the 0.002 mm that COARSE leaves the width change's chains bowed by.
COARSE_STRAIGHT_TOLERANCE = 0.01
# At spacing 0.3 the drape's first burst piles vertices up; see
# test_optimize_chains.
DRAPE_SETTINGS = replace(COARSE, spacing=0.2, settled_change=0.002)
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
    return simplify_chains(scenario.chains, run, tolerance)


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

        (result,) = simplify_chains([alone], run, WIDE_TOLERANCE)

        assert np.array_equal(
            result,
            simplify(optimised.polyline, optimised.fixed, DISC, WIDE_TOLERANCE),
        )

    def test_a_run_of_other_chains_is_refused(
        self, drape: tuple[ChainsScenario, RunResult]
    ) -> None:
        scenario, run = drape

        with pytest.raises(ValueError, match="2 optimised chains for 1"):
            simplify_chains(scenario.chains[:1], run)

    def test_a_width_change_merges_each_side_up_to_the_shared_junction(
        self, log: list[str]
    ) -> None:
        scenario = scenarios.width_change()
        run = optimized(scenario, COARSE)

        result = simplify_run(scenario, run, COARSE_STRAIGHT_TOLERANCE)

        for simplified, c in zip(result, run.chains, strict=True):
            assert np.array_equal(simplified, c.polyline[[0, -1]])
        wide, narrow = result
        assert np.array_equal(wide[-1], narrow[0])
        assert not any(line.startswith("WARNING") for line in log)


def unmoved(*lines: tuple[FloatArray, BoolArray]) -> RunResult:
    """A run that left each of `lines`, its points and fixed vertices, as
    it was."""
    return RunResult(
        tuple(
            ChainResult(points, fixed, 0, measure(points, SPACING), no_contacts(points))
            for points, fixed in lines
        ),
        Termination.CONVERGED,
        1,
    )


BUMP = 0.005
"""How far a chain's middle vertex lies off its chord, mm: within
WIDE_TOLERANCE."""
NEAR = 0.002
"""How far inside a distance kept something lies, mm."""
WIDE = 3 * PAIR_WIDTH
LARGE_CLEARANCE = 2.5 * PAIR_CLEARANCE


@pytest.mark.parametrize(
    ("width", "clearance", "beyond", "kept"),
    [
        (PAIR_WIDTH, PAIR_CLEARANCE, -NEAR, 3),
        (PAIR_WIDTH, PAIR_CLEARANCE, scenarios.PAIR_INFLATION, 2),
        (WIDE, PAIR_CLEARANCE, -NEAR, 3),
        (PAIR_WIDTH, LARGE_CLEARANCE, -NEAR, 3),
    ],
    ids=["near", "far", "wide", "large clearance"],
)
def test_a_neighbour_beside_a_chain_s_box_blocks_a_merge_within_clearance(
    width: float, clearance: float, beyond: float, kept: int
) -> None:
    # The chord lies on the box's lower side; the other track runs `beyond`
    # the distance kept from it below it, outside the box, so only its
    # clearance reaches the chord. The layer holds both kinds of track, so
    # the box must reach as far as the wider of the two distances.
    def given(
        name: str, line: FloatArray, width: float, clearance: float
    ) -> ChainInput:
        return ChainInput(
            ChainCentreline(name, name, width, clearance, line),
            "F.Cu",
            free(len(line)),
            FreeSpace(),
        )

    below = track_inflation(PAIR_WIDTH, PAIR_CLEARANCE, width, clearance) + beyond
    bumped = path((0.0, 0.0), (2.5, BUMP), (5.0, 0.0))
    other = path((0.0, -below), (5.0, -below))
    chains = [
        given("A", bumped, PAIR_WIDTH, PAIR_CLEARANCE),
        given("B", other, width, clearance),
    ]
    run = unmoved(*((c.centreline.points, c.fixed) for c in chains))

    result, _ = simplify_chains(chains, run, WIDE_TOLERANCE)

    assert len(result) == kept


BEND = 1.0
"""Radius of a coupled section's arc, mm."""
BEND_STEP = 0.02
"""rad between the arc's vertices."""
SECTION_TOLERANCE = 0.05
ROOM = 3e-3
"""mm beyond clearance: room for the outer track's corners at a few merged
vertices, less than the chord tolerance would take."""


def arc_section_beside_a_wall() -> tuple[FloatArray, ChainCentreline]:
    """A coupled section's centreline bending through 90° at `BEND`, and a
    track of another net beside its outer track at clearance."""
    angles = np.arange(0.0, np.pi / 2 + BEND_STEP / 2, BEND_STEP)
    arc = BEND * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    outside = BEND + DP_HALF_EXTENT + PAIR_CLEARANCE + PAIR_WIDTH / 2 + ROOM
    wall = ChainCentreline(
        "wall", "W", PAIR_WIDTH, PAIR_CLEARANCE, arc / BEND * outside
    )
    return arc, wall


def test_a_section_s_merged_corners_keep_clear_of_copper_outside_its_bend() -> None:
    arc, wall = arc_section_beside_a_wall()
    probe = scenarios.coupled(FreeSpace(), *(tuple(p) for p in arc))
    facing = pair_block(probe, SPACING).facing.centreline
    field = chain_obstacles(facing.net, facing.width, PAIR_CLEARANCE, [wall])
    block = pair_block(replace(probe, field=field), SPACING)
    line = block.chain.centreline.points
    fixed = np.zeros(len(line), dtype=bool)
    fixed[[0, -1]] = True
    run = RunResult(
        (ChainResult(line, fixed, 0, measure(line, SPACING), no_contacts(line)),),
        Termination.CONVERGED,
        1,
    )

    (centreline,) = simplify_chains(
        [replace(block.facing, field=field)],
        run,
        SECTION_TOLERANCE,
        sections={0: block},
    )

    assert len(centreline) < len(line)
    keep = PAIR_CLEARANCE + PAIR_WIDTH - INSIDE_TOLERANCE
    for track in block.offset(centreline):
        assert LineString(track).distance(LineString(wall.points)) >= keep


CREST_SPAN = 0.6
"""rad either side of a crest's top."""
SAG = 0.04
"""How far a chain over a crest bows up in its middle, mm: within
SECTION_TOLERANCE."""


def section_over_a_crest() -> tuple[PairBlock, FloatArray]:
    """A coupled section bending over a crest of radius `BEND`, a vertex at
    its top, and its centreline."""
    angles = np.pi / 2 + np.arange(-CREST_SPAN, CREST_SPAN + BEND_STEP / 2, BEND_STEP)
    arc = BEND * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    block = pair_block(
        scenarios.coupled(FreeSpace(), *(tuple(p) for p in arc)), SPACING
    )
    return block, block.chain.centreline.points


def track_gaps(
    block: PairBlock, centreline: FloatArray, other: FloatArray
) -> list[float]:
    """How far each of the tracks at `centreline` lies from `other`, mm."""
    return [
        LineString(track).distance(LineString(other))
        for track in block.offset(centreline)
    ]


def test_a_section_s_merged_corners_keep_clear_of_another_selected_chain() -> None:
    # The other chain lies beyond the reach of the centreline's box but
    # within that of the corners' box: the corners of the outer track at
    # the vertices kept next to merged edges reach past its optimised shape.
    block, line = section_over_a_crest()
    facing = block.facing
    reach = track_inflation(
        facing.centreline.width, facing.centreline.clearance, PAIR_WIDTH, PAIR_CLEARANCE
    )
    y = line[:, 1].max() + reach + NEAR
    wall = path((-BEND / 2, y), (BEND / 2, y))
    chains = [
        facing,
        ChainInput(
            ChainCentreline("wall", "W", PAIR_WIDTH, PAIR_CLEARANCE, wall),
            facing.layer,
            np.ones(len(wall), dtype=bool),
            FreeSpace(),
        ),
    ]

    centreline, _ = simplify_chains(
        chains,
        unmoved((line, free(len(line))), (wall, chains[1].fixed)),
        SECTION_TOLERANCE,
        sections={0: block},
    )

    assert len(centreline) < len(line)
    keep = PAIR_CLEARANCE + PAIR_WIDTH - INSIDE_TOLERANCE
    assert min(track_gaps(block, centreline, wall)) >= keep


def test_a_chain_keeps_clear_of_a_section_simplified_before_it() -> None:
    # The chain's chord clears the section's optimised tracks, but not the
    # corners its merged centreline gives them, which reach past the box
    # around the optimised tracks.
    block, line = section_over_a_crest()
    facing = block.facing
    top = max(float(track[:, 1].max()) for track in block.offset(line))
    x = np.linspace(-2 * BEND, 2 * BEND, 7)
    over = np.column_stack(
        [x, top + scenarios.PAIR_INFLATION + NEAR + SAG * (1 - (x / x[-1]) ** 2)]
    )
    chains = [
        facing,
        ChainInput(
            ChainCentreline("over", "W", PAIR_WIDTH, PAIR_CLEARANCE, over),
            facing.layer,
            free(len(over)),
            FreeSpace(),
        ),
    ]

    centreline, chord = simplify_chains(
        chains,
        unmoved((line, free(len(line))), (over, chains[1].fixed)),
        SECTION_TOLERANCE,
        sections={0: block},
    )

    assert len(centreline) < len(line)
    keep = PAIR_CLEARANCE + PAIR_WIDTH - INSIDE_TOLERANCE
    assert min(track_gaps(block, centreline, chord)) >= keep


WAVE_STEP = 0.05
KEEP_EVERY = 5
PARALLEL = 5e-13
"""The most, as 1 - cos of the angle, a segment of one track may turn from
the other's: 1e-6 rad, as the CLI's check allows."""


def test_kept_vertices_offset_into_tracks_parallel_segment_for_segment() -> None:
    # As KiCad's router expects of a pair: otherwise a later edit of it in
    # KiCad no longer treats it as coupled. A wave, unlike an arc, bends
    # unevenly, so tracks offset before merging would not be parallel.
    x = np.arange(0.0, 4.0 + WAVE_STEP / 2, WAVE_STEP)
    wave = [(float(u), float(0.5 * np.sin(u))) for u in x]
    block = pair_block(scenarios.coupled(FreeSpace(), *wave), SPACING)
    line = block.chain.centreline.points
    last = len(line) - 1
    kept = line[sorted({0, 1, *range(1, last, KEEP_EVERY), last - 1, last})]

    a, b = block.offset(kept)

    assert len(a) == len(b) == len(kept)
    along_a, along_b = (np.diff(t, axis=0) for t in (a, b))
    cosine = np.einsum("ij,ij->i", along_a, along_b) / (
        np.linalg.norm(along_a, axis=1) * np.linalg.norm(along_b, axis=1)
    )
    assert (1.0 - cosine <= PARALLEL).all()
