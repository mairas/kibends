"""Parity of the Rust energy terms, chain field, composite field and test
fields with the Python code they replace, on every scenario's start state.
Scaffolding of the port: deleted with the Python references."""

import math
from collections.abc import Sequence
from dataclasses import replace

import chain_field_reference as chain_ref
import distance_fields_reference as fields_ref
import energy_reference as ref
import numpy as np
import pytest
import scenarios

from kibends import energy
from kibends._core import testing
from kibends.energy import BoolArray, EnergyParams, FloatArray, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import CompositeField, Obstacles, chains_field

DIFFERENCES: dict[str, tuple[float, float]] = {}
"""The largest absolute and relative difference seen per quantity."""
TIES: list[int] = []
"""Points whose nearest edge ties with another, per chain-field query."""


def compare(name: str, rust: object, python: object, *, exact: bool) -> None:
    """Record how far `rust` lies from `python`, and assert they are equal
    to the bit with `exact`, else to round-off."""
    a = np.asarray(rust, dtype=float)
    b = np.asarray(python, dtype=float)
    assert a.shape == b.shape, name
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b), err_msg=name)
    finite = np.isfinite(b)
    np.testing.assert_array_equal(a[~finite], b[~finite], err_msg=name)
    delta = np.abs(a[finite] - b[finite])
    worst = float(delta.max(initial=0.0))
    scale = np.maximum(np.abs(b[finite]), 1e-300)
    relative = float((delta / scale).max(initial=0.0))
    seen = DIFFERENCES.get(name, (0.0, 0.0))
    DIFFERENCES[name] = (max(seen[0], worst), max(seen[1], relative))
    if exact:
        np.testing.assert_array_equal(a, b, err_msg=name)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=name)


def resampled(points: FloatArray, spacing: float, seed: int) -> FloatArray:
    """`points` at about `spacing`, each interior vertex moved a little."""
    arc = energy.arc_lengths(points)
    count = max(int(arc[-1] / spacing), 2)
    at: FloatArray = np.linspace(0.0, float(arc[-1]), count)
    line: FloatArray = np.column_stack(
        [np.interp(at, arc, points[:, 0]), np.interp(at, arc, points[:, 1])]
    )
    wiggle = np.random.default_rng(seed).normal(scale=spacing / 4, size=(count - 2, 2))
    line[1:-1] += wiggle
    return line


Case = tuple[str, FloatArray, BoolArray, ObstacleField, EnergyParams]


def cases() -> list[Case]:
    found: list[Case] = []
    single = [
        scenarios.wrap(),
        scenarios.channel(0.0),
        scenarios.channel(0.5),
        scenarios.topology(),
        scenarios.pad_row(),
        scenarios.starts_inside(),
        scenarios.detour(),
        scenarios.narrow_gap(),
    ]
    for s in single:
        found.append((s.name, s.polyline, s.fixed, s.field, s.params))
    several = [
        scenarios.channel_pair(),
        scenarios.uneven_channel_pair(),
        scenarios.tent(),
        scenarios.same_net_wrap(),
        scenarios.crowded(),
        scenarios.drape(),
        scenarios.squeeze(),
        scenarios.hooks(),
        scenarios.width_change(),
    ]
    for s in several:
        for c in s.chains:
            found.append(
                (
                    f"{s.name} {c.centreline.description}",
                    c.centreline.points,
                    c.fixed,
                    c.field,
                    s.params,
                )
            )
    repelling = replace(scenarios.PARAMS, repulsion=0.7, falloff_range=1.5)
    more: list[Case] = []
    for k, (name, points, _, field, params) in enumerate(found):
        line = resampled(points, 0.1, k)
        fixed = np.zeros(len(line), dtype=bool)
        fixed[[0, -1]] = True
        more.append((f"{name} resampled", line, fixed, field, params))
        more.append((f"{name} repelling", line, fixed, field, repelling))
    return found + more


CASES = cases()
IDS = [c[0] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_polyline_terms(case: Case) -> None:
    _, points, fixed, _, params = case
    for short_edge in (0.0, 0.1, 0.3):
        rust = energy.shape_energy(points, fixed, short_edge)
        python = ref.shape_energy(points, fixed, short_edge)
        compare("shape value", rust[0], python[0], exact=True)
        compare("shape gradient", rust[1], python[1], exact=True)
    for limit in (0.5, 1.8, 5.0):
        rust = energy.bend_radius_energy(points, fixed, limit, params)
        python = ref.bend_radius_energy(points, fixed, limit, params)  # pyright: ignore[reportArgumentType]
        compare("bend value", rust[0], python[0], exact=False)
        compare("bend gradient", rust[1], python[1], exact=True)
    compare(
        "arc lengths", energy.arc_lengths(points), ref.arc_lengths(points), exact=True
    )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_quadrature_terms(case: Case) -> None:
    _, points, _, field, params = case
    k = params.points_per_edge
    at = energy.quadrature_points(points, k)
    compare("quadrature points", at, ref.quadrature_points(points, k), exact=True)
    distance, direction = field.smooth_distance(at, energy.obstacle_reach(params))
    for make, make_ref, curve, curve_ref in (
        (
            energy.obstacle_integrand,
            ref.obstacle_integrand,
            energy.obstacle_curvature,
            ref.obstacle_curvature,
        ),
        (
            energy.hinge_integrand,
            ref.hinge_integrand,
            energy.hinge_curvature,
            ref.hinge_curvature,
        ),
    ):
        integrand, integrand_ref = make(params), make_ref(params)  # pyright: ignore[reportArgumentType]
        curvature, curvature_ref = curve(params), curve_ref(params)  # pyright: ignore[reportArgumentType]
        value, gradient, slopes = energy.edge_integral(
            points, distance, direction, integrand, k
        )
        value_ref, gradient_ref, slopes_ref = ref.edge_integral(
            points, distance, direction, integrand_ref, k
        )
        compare("edge integral value", value, value_ref, exact=False)
        compare("edge integral gradient", gradient, gradient_ref, exact=True)
        compare("edge integral slopes", slopes, slopes_ref, exact=True)
        compare(
            "edge curvature",
            energy.edge_curvature(points, distance, direction, integrand, curvature, k),
            ref.edge_curvature(
                points, distance, direction, integrand_ref, curvature_ref, k
            ),
            exact=True,
        )
        compare(
            "point stiffness",
            energy.point_stiffness(distance, direction, integrand, curvature),
            ref.point_stiffness(distance, direction, integrand_ref, curvature_ref),
            exact=True,
        )


SWEEP = np.array(
    [
        *(-np.inf, -2.0, -0.3, -1e-9, -0.0, 0.0, 1e-9, 0.01, 0.019999, 0.02),
        *(0.05, 0.1, 0.2, 0.2 + 1e-12, 0.7, 0.999, 1.0, 1.5, 3.0, 1e3),
        *(np.inf, np.nan),
    ]
)
"""Distances through every part of the integrands: inside, at the
clearance, within and at the standoff, within and at the falloff, and
beyond."""

PARAMS = [
    scenarios.PARAMS,
    replace(scenarios.PARAMS, repulsion=0.7, falloff_range=1.5, standoff=0.2),
    replace(scenarios.PARAMS, repulsion=0.3, falloff_range=0.2, standoff=1.0),
]


@pytest.mark.parametrize("params", PARAMS)
def test_integrands_on_a_sweep(params: EnergyParams) -> None:
    for name, make, make_ref in (
        ("obstacle", energy.obstacle_integrand, ref.obstacle_integrand),
        ("hinge", energy.hinge_integrand, ref.hinge_integrand),
        ("repulsion", energy.repulsion_integrand, ref.repulsion_integrand),
    ):
        value, slope = make(params)(SWEEP)
        value_ref, slope_ref = make_ref(params)(SWEEP)  # pyright: ignore[reportArgumentType]
        compare(f"{name} integrand", value, value_ref, exact=True)
        compare(f"{name} slope", slope, slope_ref, exact=True)
    for name, curve, curve_ref in (
        ("obstacle", energy.obstacle_curvature, ref.obstacle_curvature),
        ("hinge", energy.hinge_curvature, ref.hinge_curvature),
    ):
        compare(
            f"{name} curvature",
            curve(params)(SWEEP),
            curve_ref(params)(SWEEP),  # pyright: ignore[reportArgumentType]
            exact=True,
        )
    assert energy.obstacle_reach(params) == ref.obstacle_reach(params)  # pyright: ignore[reportArgumentType]


def test_attraction_on_a_sweep() -> None:
    params = replace(scenarios.PARAMS, attraction=0.3, attraction_range=3.0)
    r = np.array([0.0, 0.2, 0.5, 0.8, 0.81, 1.7, 2.999, 3.0, 4.0, np.inf])
    for limit in (0.0, 0.8, 3.5):
        limits = np.full(len(r), limit)
        rust = energy.attraction_integrand(params)(r, limits)
        python = ref.attraction_integrand(params)(r, limits)  # pyright: ignore[reportArgumentType]
        compare("attraction value", rust[0], python[0], exact=True)
        compare("attraction slope", rust[1], python[1], exact=True)


def test_degenerate_and_reversed_polylines() -> None:
    params = PARAMS[1]
    shapes = [
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0 + 5e-7, 0.0], [2.0, 0.5]]),
        np.array([[0.0, 0.0], [2.0, 0.0], [0.1, 0.05], [0.0, 1.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        np.array([[-2.0, 1.1], [0.0, 1.1], [0.0, 1.1], [2.0, 1.2], [2.0, 1.2]]),
    ]
    for points in shapes:
        fixed = np.zeros(len(points), dtype=bool)
        for short_edge in (0.0, 0.1):
            rust = energy.shape_energy(points, fixed, short_edge)
            python = ref.shape_energy(points, fixed, short_edge)
            compare("shape value", rust[0], python[0], exact=True)
            compare("shape gradient", rust[1], python[1], exact=True)
        rust = energy.bend_radius_energy(points, fixed, 1.8, params)
        python = ref.bend_radius_energy(points, fixed, 1.8, params)  # pyright: ignore[reportArgumentType]
        compare("bend value", rust[0], python[0], exact=False)
        compare("bend gradient", rust[1], python[1], exact=True)


def test_combine_fields_and_smooth_minimum() -> None:
    rng = np.random.default_rng(0)
    for count in (1, 2, 3):
        distances = [rng.uniform(-0.05, 0.1, 50) for _ in range(count)]
        distances[0][::7] = np.inf
        if count > 1:
            distances[1][::3] = np.inf
        smooth, shares = energy.combine_fields(*distances)
        smooth_ref, shares_ref = ref.combine_fields(*distances)
        compare("combined", smooth, smooth_ref, exact=True)
        compare("combined shares", np.array(shares), np.array(shares_ref), exact=True)
    point = rng.integers(0, 20, 80).astype(np.intp)
    distance = rng.uniform(0.0, 0.08, 80)
    smooth, share = energy.smooth_minimum(point, distance, 20)
    smooth_ref, share_ref = ref.smooth_minimum(point, distance, 20)
    compare("smooth minimum", smooth, smooth_ref, exact=True)
    compare("smooth shares", share, share_ref, exact=True)
    direction = rng.normal(size=(80, 2))
    compare(
        "weighted sum",
        energy.weighted_sum(point, share, direction, 20),
        ref.weighted_sum(point, share_ref, direction, 20),
        exact=True,
    )
    lengths = rng.uniform(0.0, 1.0, 9)
    half_turns = rng.normal(size=8)
    half_turns[3] = 0.0
    half_turns[4] = 0.0
    rust = energy.edge_bend_radii(lengths, half_turns)
    python = ref.edge_bend_radii(lengths, half_turns)
    compare("bend radii", rust[0], python[0], exact=True)
    compare("bend spread", rust[1], python[1], exact=True)


def field_pairs() -> list[tuple[str, ObstacleField, ObstacleField]]:
    discs = [(0.0, 0.0, 1.0), (5.0, 1.0, 0.8), (5.0, -1.0, 0.8)]
    boxes = [
        (6.5, 3.5, 2.0, 2.0, scenarios.ENVELOPE),
        (2.5, -2.1, 1.0, 2.9, 0.0),
        (5.0, 1.5, 3.0, 0.1, 0.0),
    ]
    pairs: list[tuple[str, ObstacleField, ObstacleField]] = [
        ("free", testing.FreeSpace(), fields_ref.FreeSpace())
    ]
    for d in discs:
        pairs.append((f"disc {d}", testing.Disc(*d), fields_ref.Disc(*d)))
    for b in boxes:
        pairs.append((f"box {b}", testing.Box(*b), fields_ref.Box(*b)))
    for gap in (0.1, 1.0, 5.0):
        pairs.append((f"walls {gap}", testing.Walls(gap), fields_ref.Walls(gap)))
        pairs.append(
            (
                f"misleading {gap}",
                testing.MisleadingWalls(gap),
                fields_ref.MisleadingWalls(gap),
            )
        )
    pairs.append(
        (
            "union",
            testing.Union(
                [
                    testing.Disc(*discs[1]),
                    testing.Disc(*discs[2]),
                    testing.Box(*boxes[0]),
                ]
            ),
            fields_ref.Union(
                [
                    fields_ref.Disc(*discs[1]),
                    fields_ref.Disc(*discs[2]),
                    fields_ref.Box(*boxes[0]),
                ]
            ),
        )
    )
    pairs.append(
        (
            "union with walls",
            testing.Union([testing.Walls(1.0), testing.Disc(*discs[0])]),
            fields_ref.Union([fields_ref.Walls(1.0), fields_ref.Disc(*discs[0])]),
        )
    )
    return pairs


FIELD_PAIRS = field_pairs()


@pytest.mark.parametrize(
    ("name", "rust", "python"), FIELD_PAIRS, ids=[p[0] for p in FIELD_PAIRS]
)
def test_test_fields_match_their_python_originals(
    name: str, rust: ObstacleField, python: ObstacleField
) -> None:
    rng = np.random.default_rng(1)
    on_cases = [energy.quadrature_points(c[1], 3) for c in CASES[:20]]
    points = np.concatenate([rng.uniform(-3.0, 9.0, (400, 2)), *on_cases])
    for method in ("signed_distance", "smooth_distance"):
        d, g = getattr(rust, method)(points)
        d_ref, g_ref = getattr(python, method)(points)
        compare(f"{name} {method}", d, d_ref, exact=True)
        compare(f"{name} {method} gradient", g, g_ref, exact=True)
    starts = points[:-1]
    ends = points[1:] + rng.normal(scale=0.5, size=(len(points) - 1, 2))
    for tolerance in (0.0, 0.05):
        np.testing.assert_array_equal(
            rust.segments_clear(starts, ends, tolerance),
            python.segments_clear(starts, ends, tolerance),
        )


def several_chains() -> list[tuple[str, Sequence[ChainInput]]]:
    return [
        (s.name, s.chains)
        for s in (
            scenarios.channel_pair(),
            scenarios.uneven_channel_pair(),
            scenarios.tent(),
            scenarios.crowded(),
            scenarios.drape(),
            scenarios.squeeze(),
            scenarios.hooks(),
            scenarios.width_change(),
        )
    ]


@pytest.mark.parametrize("scenario", several_chains(), ids=lambda s: s[0])
def test_chain_fields_match_the_python_one(
    scenario: tuple[str, Sequence[ChainInput]],
) -> None:
    _, chains = scenario
    inflated = [(c.centreline, 0.2 + 0.05 * k) for k, c in enumerate(chains)]
    rust = chains_field(inflated)
    python = chain_ref.chains_field(inflated)
    rng = np.random.default_rng(2)
    at = np.concatenate(
        [energy.quadrature_points(c.centreline.points, 3) for c in chains]
        + [rng.uniform(-1.0, 21.0, (300, 2)) * [1.0, 0.3]]
    )
    for reach in (0.3, 1.0, math.inf):
        d, g, edge = rust.nearest(at, reach)
        d_ref, g_ref, edge_ref = python.nearest(at, reach)
        compare("chain signed", d, d_ref, exact=True)
        # Of equally near edges the last pair answers; scipy's tree listed
        # pairs in its own order, the Rust tree by edge.
        tied = edge != edge_ref
        TIES.append(int(tied.sum()))
        compare("chain signed gradient", g[~tied], g_ref[~tied], exact=True)
        found = rust.smooth_answers(at, reach)
        found_ref = python.smooth_answers(at, reach)
        parts = ("smooth", "gradient", "point", "edge", "share", "direction")
        for part, got, expected in zip(parts, found, found_ref, strict=True):
            compare(f"chain {part}", got, expected, exact=True)
    ends = at + rng.normal(scale=0.3, size=at.shape)
    for tolerance in (0.0, 0.05):
        np.testing.assert_array_equal(
            rust.segments_clear(at, ends, tolerance),
            python.segments_clear(at, ends, tolerance),
        )
    edges = rust.static_edges
    edges_ref = python.static_edges
    compare(
        "chain static inflations", edges.inflations, edges_ref.inflations, exact=True
    )


def test_composites_match_the_python_combination() -> None:
    walls = testing.Walls(1.0)
    disc = testing.Disc(5.0, 0.5, 0.3)
    chains = chains_field(
        [
            (scenarios.channel_pair().chains[0].centreline, 0.4),
            (scenarios.channel_pair().chains[1].centreline, 0.4),
        ]
    )
    board = Obstacles(())
    rust = CompositeField([walls, disc, chains, board])
    assert rust.core is not None
    python_fields: list[ObstacleField] = [
        fields_ref.Walls(1.0),
        fields_ref.Disc(5.0, 0.5, 0.3),
        chains,
        board,
    ]
    rng = np.random.default_rng(3)
    at = rng.uniform(-1.0, 11.0, (500, 2)) * [1.0, 0.25]
    distance = np.full(len(at), np.inf)
    gradient = np.zeros((len(at), 2))
    for f in python_fields:
        d, g = f.signed_distance(at, 2.0)
        nearer = d < distance
        distance = np.where(nearer, d, distance)
        gradient = np.where(nearer[:, None], g, gradient)
    d, g = rust.signed_distance(at, 2.0)
    compare("composite signed", d, distance, exact=True)
    compare("composite signed gradient", g, gradient, exact=True)
    answers = [f.smooth_distance(at, 2.0) for f in python_fields]
    smooth, shares = ref.combine_fields(*(a[0] for a in answers))
    total = np.zeros((len(at), 2))
    for share, (_, g_part) in zip(shares, answers, strict=True):
        total += share[:, None] * g_part
    d, g = rust.smooth_distance(at, 2.0)
    compare("composite smooth", d, smooth, exact=True)
    compare("composite smooth gradient", g, total, exact=True)
    mixed = CompositeField([walls, fields_ref.Disc(5.0, 0.5, 0.3)])
    assert mixed.core is None
    d, _ = mixed.smooth_distance(at, 2.0)
    d_all, _ = CompositeField([walls, disc]).smooth_distance(at, 2.0)
    compare("composite fallback", d, d_all, exact=True)


def test_nan_field_is_nan() -> None:
    d, _ = testing.NanField().smooth_distance(np.zeros((3, 2)))
    assert np.isnan(d).all()


def test_report_differences() -> None:
    """Prints the largest differences seen, run last with -s."""
    for name, (worst, relative) in sorted(DIFFERENCES.items()):
        print(f"{name:40s} abs {worst:.3g} rel {relative:.3g}")
    print(f"chain-field points answered by another of tied edges: {sum(TIES)}")
