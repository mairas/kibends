# pyright: basic
"""Parity of the Rust static field with the Python one it replaces: the
same distances within NM_ROUNDING, the same pairs found, the same checks.
Scaffolding of the port; deleted with the Python field."""

import math

import numpy as np
import pytest
import shapely
from conftest import FakeBoard, kipy_pad, kipy_track, kipy_via, layer_snapshot
from scenarios import (
    channel_pair,
    crowded,
    drape,
    hooks,
    same_net_wrap,
    squeeze,
    tent,
    uneven_channel_pair,
    width_change,
)
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from kibends import _core
from kibends.energy import FloatArray, IntArray, smooth_reach
from kibends.obstacles import (
    ChainField,
    CompositeField,
    Found,
    Obstacle,
    Obstacles,
    StaticItems,
    build_obstacles,
    candidate_pairs,
    chain_inflation,
    chains_field,
    complex_points,
    containment,
    near_segments,
    nearest_of_pairs,
    pair_depths,
    pairs_inside,
    plain_shapes,
    smooth_of_pairs,
    static_distances,
    static_smooth,
)
from kibends.pairs import diff_pair_nets
from kibends.snapshot import ViaZone
from kibends.vias import _depth  # pyright: ignore[reportPrivateUsage]

NM_ROUNDING = 2e-6
"""Within this the Rust field's distances must equal the Python field's."""

worst: dict[str, float] = {}
"""The largest difference each comparison found, printed with -s."""


def _record(name: str, a: object, b: object) -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, name
    same_inf = (a == b) | (np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b)))
    finite = np.isfinite(a) & np.isfinite(b)
    assert (same_inf | finite).all(), name
    gap = float(np.max(np.abs(a[finite] - b[finite]), initial=0.0))
    worst[name] = max(worst.get(name, 0.0), gap)
    print(f"{name}: max abs difference {worst[name]:.3g}")
    assert gap <= NM_ROUNDING, name


def rust_of(field: Obstacles) -> _core.Obstacles:
    items = field.items
    outline = field.board_outline
    return _core.Obstacles(
        plain_shapes(o.geometry for o in items),
        np.array([o.inflation for o in items], dtype=float),
        [o.net for o in items],
        np.array([o.extent for o in items], dtype=float),
        [o.layer for o in items],
        None if outline is None else plain_shapes([outline]),
        field.edge_inflation,
    )


SCENE_OUTLINE = box(-3.0, -3.0, 9.0, 4.0)
SCENE = [
    Obstacle("rectangular pad", box(0.0, 0.0, 1.0, 0.6), 0.3),
    Obstacle("round pad", Point(3.0, 0.3).buffer(0.4), 0.25),
    Obstacle(
        "copper with a hole",
        Polygon(
            box(5.0, -1.0, 7.0, 1.0).exterior.coords,
            [box(5.5, -0.5, 6.5, 0.5).exterior.coords],
        ),
        0.2,
    ),
    Obstacle("track", LineString([(0.0, 2.0), (4.0, 2.5)]), 0.35),
    Obstacle("via", Point(6.0, 2.5), 0.5),
    Obstacle("logo", Point(1.0, -2.0).buffer(0.8, quad_segs=256), 0.2),
]


def _random_copper(rng: np.random.Generator) -> list[Obstacle]:
    items: list[Obstacle] = []
    for k in range(int(rng.integers(3, 15))):
        c = rng.uniform(0.0, 6.0, 2)
        kind = int(rng.integers(0, 6))
        inflation = float(rng.choice([0.0, 0.1, 0.25, 0.4, 0.6]))
        geometry: BaseGeometry
        if kind == 0:
            w, h = rng.uniform(0.05, 1.0, 2)
            geometry = box(c[0], c[1], c[0] + w, c[1] + h)
        elif kind == 1:
            geometry = Point(*c).buffer(float(rng.uniform(0.05, 0.6)), 8)
        elif kind == 2:
            geometry = LineString([c, *(c + rng.uniform(-2.0, 2.0, (2, 2)))])
        elif kind == 3:
            geometry = Point(*c)
        elif kind == 4:
            geometry = Point(*c).buffer(1.0, quad_segs=64)
        else:
            geometry = Polygon(
                box(c[0], c[1], c[0] + 1.5, c[1] + 1.5).exterior.coords,
                [box(c[0] + 0.4, c[1] + 0.4, c[0] + 1.1, c[1] + 1.1).exterior.coords],
            )
        items.append(Obstacle(f"item {k}", geometry, inflation, "N", 0.1, "F.Cu"))
    items.append(Obstacle("again", items[0].geometry, items[0].inflation))
    return items


def _fields() -> list[tuple[str, Obstacles]]:
    rng = np.random.default_rng(5)
    holed_board = Polygon(
        box(-0.5, -0.5, 6.5, 6.5).exterior.coords,
        [box(2.8, 2.8, 3.2, 3.2).exterior.coords],
    )
    fields = [
        ("scene", Obstacles(SCENE, SCENE_OUTLINE, 0.4)),
        ("scene without edge", Obstacles(SCENE)),
        ("edge alone", Obstacles([], SCENE_OUTLINE, 0.4)),
        ("round edge", Obstacles([], box(0, 0, 6, 4).buffer(1.0, quad_segs=64), 0.3)),
    ]
    for trial in range(8):
        outline = holed_board if trial % 2 else box(-0.5, -0.5, 6.5, 6.5)
        fields.append(
            (
                f"random {trial}",
                Obstacles(_random_copper(rng), outline if trial % 3 else None, 0.3),
            )
        )
    fake = FakeBoard(
        pads=[
            kipy_pad("p1", "GND", 0, 0),
            kipy_pad("p2", "VCC", 2, 0),
            kipy_pad("tht", "GND", 4, 0, drill=0.8),
        ],
        tracks=[
            kipy_track("narrow", "GND", (0, 2), (2, 2), width=0.25),
            kipy_track("wide", "GND", (2, 2), (4, 2), width=0.5),
            kipy_track("other", "VCC", (5, 2), (7, 2), width=0.5),
            kipy_track("sig", "SIG", (0, 4), (7, 4), width=0.25),
        ],
        vias=[kipy_via("v1", "GND", 6, 0)],
    )
    fields.append(("board", build_obstacles(layer_snapshot(fake, "F.Cu"), "SIG", 0.25)))
    return fields


FIELDS = _fields()
FIELD_IDS = [name for name, _ in FIELDS]


def _grid(field: Obstacles) -> FloatArray:
    bounds = shapely.total_bounds(
        [o.geometry for o in field.items]
        + ([] if field.board_outline is None else [field.board_outline])
    )
    rng = np.random.default_rng(1)
    x0, y0, x1, y1 = bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1
    x, y = np.meshgrid(np.linspace(x0, x1, 23), np.linspace(y0, y1, 19))
    grid = np.column_stack([x.ravel(), y.ravel()])
    vertices = shapely.get_coordinates(
        [o.geometry for o in field.items]
        + ([] if field.board_outline is None else [field.board_outline.boundary])
    )
    along = vertices[: len(vertices) - 1] * 0.5 + vertices[1:] * 0.5
    return np.vstack([grid, rng.uniform((x0, y0), (x1, y1), (300, 2)), vertices, along])


@pytest.mark.parametrize("name,field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("reach", [0.3, math.inf])
def test_signed_distance(name: str, field: Obstacles, reach: float) -> None:
    points = _grid(field)
    rust = rust_of(field)

    d, g = rust.signed_distance(points, reach)

    expected, toward = field.signed_distance(points, reach)
    _record("signed_distance", d, expected)
    finite = np.isfinite(d)
    _record("signed_distance gradient", g[finite], toward[finite])


@pytest.mark.parametrize("name,field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("reach", [0.3, math.inf])
def test_smooth_distance(name: str, field: Obstacles, reach: float) -> None:
    points = _grid(field)

    d, g = rust_of(field).smooth_distance(points, reach)

    expected, toward = field.smooth_distance(points, reach)
    _record("smooth_distance", d, expected)
    _record("smooth_distance gradient", g, toward)


def _found_set(found: Found) -> set[tuple[int, int, int, complex, complex, bool, bool]]:
    return {
        (int(p), int(o), int(i), complex(s), complex(e), bool(a), bool(u))
        for p, s, e, o, i, a, u in zip(
            found.point,
            found.starts,
            found.ends,
            found.obstacle,
            found.item,
            found.inside,
            found.unsure,
            strict=True,
        )
    }


@pytest.mark.parametrize("name,field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("reach", [0.3, math.inf])
@pytest.mark.parametrize("slack", [0.0, 0.05])
def test_search_and_static_smooth(
    name: str, field: Obstacles, reach: float, slack: float
) -> None:
    points = _grid(field)
    rust = rust_of(field)
    search = smooth_reach(reach)

    got = Found(*_core.search_parts([(rust.static_edges, 0)], points, search, slack))

    expected = field.static_edges.search(points, search, slack)
    assert _found_set(got) == _found_set(expected)
    moved = points + np.random.default_rng(2).uniform(-0.7, 0.7, points.shape) * slack
    query = complex_points(moved)
    d, g = _core.static_smooth(query, *got, _core.StaticItems([rust.static_edges]))
    want, toward = static_smooth(query, expected, StaticItems([field.static_edges]))
    _record("static_smooth", d, want)
    _record("static_smooth gradient", g, toward)


@pytest.mark.parametrize("name,field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("below", [0.3, -0.05, math.inf])
def test_static_distances(name: str, field: Obstacles, below: float) -> None:
    points = _grid(field)
    rust = rust_of(field)

    point, obstacle, distance = _core.static_distances(
        [rust.static_edges], points, below
    )

    expected = static_distances([field.static_edges], points, below)
    got = dict(
        zip(zip(point.tolist(), obstacle.tolist(), strict=True), distance, strict=True)
    )
    want = dict(
        zip(
            zip(expected.point.tolist(), expected.obstacle.tolist(), strict=True),
            expected.distance,
            strict=True,
        )
    )
    if math.isfinite(below):
        assert got.keys() == want.keys()
    else:
        # "At least each obstacle within SMOOTH_REACH of the nearest": of
        # equally near items shapely's tree picks any, Rust the least
        # inflated, whose search is the widest.
        assert want.keys() <= got.keys()
    _record("static_distances", [got[k] for k in want], list(want.values()))


@pytest.mark.parametrize("name,field", FIELDS, ids=FIELD_IDS)
def test_obstacle_numbering(name: str, field: Obstacles) -> None:
    rust = rust_of(field).static_edges
    python = field.static_edges

    assert rust.obstacles.tolist() == python.obstacles.tolist()
    assert rust.clearance_obstacles.tolist() == python.clearance_obstacles.tolist()
    assert rust.obstacle_count == python.obstacle_count


@pytest.mark.parametrize("name,field", FIELDS, ids=FIELD_IDS)
@pytest.mark.parametrize("tolerance", [0.0, 0.01, -0.01])
def test_segments_clear(name: str, field: Obstacles, tolerance: float) -> None:
    rng = np.random.default_rng(4)
    starts = _grid(field)
    ends = starts + rng.uniform(-1.0, 1.0, starts.shape)
    ends[::7] = starts[::7]

    got = rust_of(field).segments_clear(starts, ends, tolerance)

    expected = field.segments_clear(starts, ends, tolerance)
    # shapely's tree query skips a segment of no length near a polygon; Rust
    # measures it like any other, as `ChainField` does.
    some_length = (starts != ends).any(axis=1)
    assert got[some_length].tolist() == expected[some_length].tolist()


def test_a_segment_of_no_length_is_checked_like_any_other() -> None:
    field = rust_of(Obstacles([Obstacle("pad", box(0.0, 0.0, 1.0, 1.0), 0.3)]))
    at = np.array([[0.5, -0.1], [0.5, -0.5]])

    assert field.segments_clear(at, at).tolist() == [False, True]


def test_containment_and_zone_depth() -> None:
    region = Polygon(
        box(-2, -1, 2, 1).exterior.coords, [box(-0.5, -0.5, 0.5, 0.5).exterior.coords]
    )
    zone = ViaZone("Z", {"F.Cu": region}, 0.2, 0.1)
    field = containment([zone])
    assert isinstance(field, CompositeField)
    (part,) = field.fields
    assert isinstance(part, Obstacles)
    outline = part.board_outline
    assert outline is not None
    points = _grid(part)

    d, _ = rust_of(part).signed_distance(points)
    depth = _core.zone_depth(plain_shapes([outline]), points)

    expected, _ = part.signed_distance(points)
    _record("containment", d, expected)
    _record("zone depth", depth, [_depth(outline, Point(p)) for p in points])
    assert _core.zone_depth(plain_shapes([Polygon()]), points[:1])[0] == -math.inf


def _chain_fields() -> list[ChainField]:
    fields: list[ChainField] = []
    for scenario in (
        channel_pair(),
        uneven_channel_pair(),
        tent(),
        same_net_wrap(),
        crowded(),
        drape(),
        squeeze(),
        hooks(),
        width_change(),
    ):
        chains = [c.centreline for c in scenario.chains]
        for me in chains:
            ignored = diff_pair_nets(me.net)
            fields.append(
                chains_field(
                    (o, chain_inflation(ignored, me.width, me.clearance, o))
                    for o in chains
                    if o is not me
                )
            )
    return fields


def _edges_of(
    field: ChainField,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    starts = field._starts  # pyright: ignore[reportPrivateUsage]
    ends = starts + field._along  # pyright: ignore[reportPrivateUsage]
    return (
        np.column_stack([starts.real, starts.imag]),
        np.column_stack([ends.real, ends.imag]),
        field._inflations,  # pyright: ignore[reportPrivateUsage]
        field._obstacles,  # pyright: ignore[reportPrivateUsage]
    )


def test_chain_fields_of_every_scenario() -> None:
    rng = np.random.default_rng(6)
    for field in _chain_fields():
        starts, ends, inflations, obstacles = _edges_of(field)
        if not len(starts):
            continue
        rust = _core.StaticEdges.from_segments(starts, ends, inflations, obstacles)
        lo, hi = starts.min(axis=0) - 1.0, starts.max(axis=0) + 1.0
        points = np.vstack([rng.uniform(lo, hi, (200, 2)), starts, (starts + ends) / 2])
        for reach in (0.3, math.inf):
            for slack in (0.0, 0.05):
                got = Found(*_core.search_parts([(rust, 0)], points, reach, slack))
                expected = field.static_edges.search(points, reach, slack)
                assert _found_set(got) == _found_set(expected)
                query = complex_points(points)
                d, g = _core.static_smooth(query, *got, _core.StaticItems([rust]))
                want, toward = static_smooth(
                    query, expected, StaticItems([field.static_edges])
                )
                _record("chain static_smooth", d, want)
                _record("chain static_smooth gradient", g, toward)


def _pairs(
    rng: np.random.Generator, n: int, m: int
) -> tuple[
    FloatArray, IntArray, FloatArray, FloatArray, FloatArray, FloatArray, IntArray
]:
    query = rng.uniform(0, 5, (n, 2))
    point = np.sort(rng.integers(0, n, m)).astype(np.intp)
    starts = rng.uniform(0, 5, (m, 2))
    along = rng.uniform(-1, 1, (m, 2)) * (rng.uniform(size=(m, 1)) > 0.1)
    inflations = rng.uniform(0, 0.5, m)
    limit = rng.uniform(0.5, 3.0, m)
    obstacle = rng.integers(0, 6, m).astype(np.intp)
    return query, point, starts, along, inflations, limit, obstacle


def test_pair_helpers() -> None:
    rng = np.random.default_rng(8)
    for _ in range(20):
        query, point, starts, along, inflations, limit, obstacle = _pairs(rng, 30, 300)
        order = np.lexsort((np.arange(len(point)), obstacle, point))
        point, starts, along = point[order], starts[order], along[order]
        inflations, limit, obstacle = inflations[order], limit[order], obstacle[order]
        q, s, a = complex_points(query), complex_points(starts), complex_points(along)

        got = _core.smooth_of_pairs(q, point, s, a, inflations, limit, obstacle)
        expected = smooth_of_pairs(q, point, s, a, inflations, limit, obstacle)
        _record("smooth_of_pairs", got[0], expected.distance)
        _record("smooth_of_pairs gradient", got[1], expected.gradient)
        assert got[2].tolist() == expected.point.tolist()
        assert got[3].tolist() == expected.edge.tolist()
        _record("smooth_of_pairs share", got[4], expected.share)

        near = _core.nearest_of_pairs(q, point, s, a, inflations, limit)
        want = nearest_of_pairs(q, point, s, a, inflations, limit)
        _record("nearest_of_pairs", near[0], want.distance)
        _record("nearest_of_pairs gradient", near[1], want.gradient)

        kept = _core.candidate_pairs(q, point, s, a, inflations, limit, 0.05)
        assert (
            kept.tolist()
            == candidate_pairs(q, point, s, a, inflations, limit, 0.05).tolist()
        )

        segment_a = complex_points(query[point])
        segment_ab = complex_points(rng.uniform(-1, 1, (len(point), 2)))
        assert (
            _core.pairs_inside(segment_a, segment_ab, s, a, inflations, 0.01).tolist()
            == pairs_inside(segment_a, segment_ab, s, a, inflations, 0.01).tolist()
        )
        _record(
            "pair_depths",
            _core.pair_depths(segment_a, segment_ab, s, a, inflations),
            pair_depths(segment_a, segment_ab, s, a, inflations),
        )

        ends = starts + along
        found = _core.near_segments(starts, ends, query, 0.4)
        expected_near = near_segments(starts, ends, query, 0.4)
        assert sorted(zip(*(x.tolist() for x in found), strict=True)) == sorted(
            zip(*(x.tolist() for x in expected_near), strict=True)
        )


def test_edge_cases() -> None:
    """A point on an edge, at a vertex and in a hole, and two equally near
    obstacles."""
    holed = Polygon(box(0, 0, 4, 4).exterior.coords, [box(1, 1, 3, 3).exterior.coords])
    field = Obstacles(
        [
            Obstacle("holed", holed, 0.1),
            Obstacle("left", LineString([(10, -1), (10, 1)]), 0.0),
            Obstacle("right", LineString([(12, -1), (12, 1)]), 0.0),
        ]
    )
    points = np.array([[2.0, 0.0], [0.0, 0.0], [2.0, 2.0], [1.0, 2.0], [11.0, 0.0]])
    rust = rust_of(field)
    for reach in (0.3, math.inf):
        d, g = rust.signed_distance(points, reach)
        expected, toward = field.signed_distance(points, reach)
        _record("edge cases", d, expected)
        _record("edge cases gradient", g, toward)
        s, t = rust.smooth_distance(points, reach)
        want, toward = field.smooth_distance(points, reach)
        _record("edge cases smooth", s, want)
        _record("edge cases smooth gradient", t, toward)
    (between,), _ = rust.smooth_distance(points[-1:])
    assert between == pytest.approx(1.0 - 0.005 * math.log(2), abs=1e-12)


def test_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Prints the largest differences found, for the record."""
    with capsys.disabled():
        for name, gap in sorted(worst.items()):
            print(f"parity {name}: {gap:.3g}")
