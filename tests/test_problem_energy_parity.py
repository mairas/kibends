"""Temporary: the Rust problem energy against the Python one it replaces,
on every scenario and the benchmark's bus, joint and return-via
geometries. Deleted with the Python implementation."""

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
import test_optimize_contacts
import test_optimize_vias
import test_shared_search
from conftest import CommitBoard, FakeKiCad, kipy_pad, kipy_track, return_via_board
from distance_fields import FreeSpace
from kipy.board_types import Track
from numpy.typing import ArrayLike
from scenarios import ChainsScenario, PairsScenario, Scenario, ViasScenario
from shapely.geometry import box

import kibends.optimize as optimize_module
from kibends import _core, cli
from kibends import run as pipeline
from kibends.bursts import (
    DEEPER_ROUND_OFF,
    DEFAULT_SETTINGS,
    INSIDE_TOLERANCE,
    OptimizerSettings,
)
from kibends.energy import DistanceField, EnergyParams, FloatArray
from kibends.members import Member
from kibends.neighbours import ChainInput
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE, ChainCentreline, core_field
from kibends.pair_blocks import PairBlock, PairInput, pair_block
from kibends.pairs import diff_pair_nets
from kibends.seen import ViaSight
from kibends.snapshot import ViaZone
from kibends.via_moves import MovingVias, newly_inside
from kibends.vias import StaticVia, ViaInput

Problem = optimize_module._Problem  # pyright: ignore[reportPrivateUsage]
Placed = optimize_module._Placed  # pyright: ignore[reportPrivateUsage]
AnyScenario = Scenario | ChainsScenario | PairsScenario | ViasScenario

REPORT = os.environ.get("PARITY_REPORT")


def _points(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=float).reshape(-1, 2)


def _field(field: DistanceField) -> _core.Field:
    core = core_field(field)
    assert core is not None
    return core


def core_member(member: Member) -> _core.Member:
    if isinstance(member, PairBlock):
        given = member.given
        offset = _core.PairOffset(
            given.gap,
            given.a.width,
            member.a_on_right,
            member.held,
            _points(member.pins.a_at),
            _points(member.pins.a),
            _points(member.pins.b_at),
            _points(member.pins.b),
            _points(given.a.points),
            _points(given.b.points),
        )
        facing = member.facing.centreline
        return _core.Member(
            given.layer,
            (facing.net, facing.width, facing.clearance),
            sorted(member.nets),
            [
                (t.centreline.net, t.centreline.width, t.centreline.clearance)
                for t in member.tracks
            ],
            _field(member.chain.field),
            member.chain.centreline.width,
            [_field(f) for _, f in member.chain.contacts],
            offset,
            member.limit,
        )
    c = member.centreline
    return _core.Member(
        member.layer,
        (c.net, c.width, c.clearance),
        sorted(diff_pair_nets(c.net)),
        [(c.net, c.width, c.clearance)],
        _field(member.field),
        c.width,
        [_field(f) for _, f in member.contacts],
    )


def _dicts(matrix: FloatArray) -> list[dict[int, float]]:
    return [
        {j: float(v) for j, v in enumerate(row) if not math.isnan(v)} for row in matrix
    ]


def core_sight(sight: ViaSight, members: int) -> _core.ViaSight:
    return _core.ViaSight(_dicts(sight.rings), _dicts(sight.apart), members)


def core_problem(problem: Problem) -> _core.Problem:
    count = len(problem.members)
    return _core.Problem(
        _core.Members([core_member(m) for m in problem.members]),
        [dict(r) for r in problem.rings],
        [via.given.via.diameter for via in problem.vias],
        [_field(via.given.field) for via in problem.vias],
        [
            None if via.given.stops is None else _field(via.given.stops)
            for via in problem.vias
        ],
        core_sight(problem.via_repel, count),
        core_sight(problem.via_stop, count),
        None if problem.pairs is None else problem.pairs.core,
        problem.params,
        problem.settings.short_edge,
    )


class Diff:
    def __init__(self) -> None:
        self.abs = 0.0
        self.rel = 0.0
        self.bitwise = True
        self.values = 0

    def add(self, got: ArrayLike, want: ArrayLike) -> None:
        g = np.asarray(got, dtype=float).ravel()
        w = np.asarray(want, dtype=float).ravel()
        assert g.shape == w.shape
        self.values += len(g)
        same = (g == w) | (np.isnan(g) & np.isnan(w))
        if same.all():
            return
        self.bitwise = False
        d = np.abs(g - w)[~same]
        scale = np.maximum(np.abs(w), np.abs(g))[~same]
        self.abs = max(self.abs, float(d.max()))
        self.rel = max(
            self.rel, float((d / np.where(scale > 0, scale, 1.0)).max(initial=0.0))
        )


def _positions(problem: Problem, positions: Sequence[FloatArray]) -> FloatArray:
    return _points(np.array(positions, dtype=float))


def check_problem(problem: Problem, name: str) -> dict[str, Diff]:
    core = core_problem(problem)
    diffs = {k: Diff() for k in ("total", "objective", "stiffness", "checks", "free")}
    rng = np.random.default_rng(3)
    masks = problem.energy_masks(problem.masks)
    states = [
        (problem.points, problem.positions),
        (
            [p + rng.normal(scale=0.02, size=p.shape) for p in problem.points],
            [p + rng.normal(scale=0.02, size=2) for p in problem.positions],
        ),
    ]
    for polylines, positions in states:
        seen = problem.seeing(polylines)
        core_seen = core.seen(polylines)
        want = problem.total(polylines, positions, problem.masks, seen)
        got = core.total(
            polylines,
            _positions(problem, positions),
            masks,
            problem.contact_of,
            core_seen,
        )
        diffs["total"].add(got[0], want[0])
        for g, w in zip(got[1], want[1], strict=True):
            diffs["total"].add(g, w)
        diffs["total"].add(got[2], want[2])
        # The checks: what the members see, and the vias' free distances.
        from kibends.members import shown_flat

        flat = shown_flat(problem.members, polylines, positions)
        core_flat = core.shown_flat(polylines, _positions(problem, positions))
        diffs["checks"].add(core_flat, flat)
        for below in (-INSIDE_TOLERANCE, -0.02):
            for g, w in zip(
                core_seen.distances(polylines, flat, below),
                seen.distances(polylines, flat, below),
                strict=True,
            ):
                diffs["checks"].add(g, w)
        for tolerance in (0.0, INSIDE_TOLERANCE):
            for g, w in zip(
                core_seen.segments_clear(polylines, flat, tolerance),
                seen.segments_clear(polylines, flat, tolerance),
                strict=True,
            ):
                assert (g.clear == w.clear).all()
                assert g.edge.tolist() == w.edge.tolist()
                assert g.vertex.tolist() == w.vertex.tolist()
                assert g.obstacle.tolist() == w.obstacle.tolist()
                diffs["checks"].add(g.depth, w.depth)
        assert core_seen.obstacle.tolist() == seen.obstacle.tolist()
        if problem.vias:
            check_free(problem, core, polylines, positions, diffs["free"])
    layout = problem.layout(problem.points)
    seen = problem.seeing(layout.starts)
    lookups = problem.lookups(seen)
    want_stiffness = problem.stiffness(layout, seen, lookups)
    objective = problem.objective(layout, seen, lookups)
    core_seen = core.seen(layout.starts)
    core_lookups = core.lookups(core_seen)
    per_member, per_via = core.stiffness(
        layout.starts, _positions(problem, layout.positions), core_seen, core_lookups
    )
    stiffness = optimize_module._length_stiffness(layout)  # pyright: ignore[reportPrivateUsage]
    for per_vertex, index in zip(per_member, layout.index, strict=True):
        free = index >= 0
        np.add.at(stiffness, index[free], per_vertex[free])
    stiffness[layout.via_index] += per_via
    diffs["stiffness"].add(stiffness, want_stiffness)
    core_objective = core.objective(
        layout.starts,
        layout.index,
        layout.offsets,
        layout.via_index,
        layout.count,
        masks,
        problem.contact_of,
        core_seen,
        core_lookups,
    )
    x = layout.x0.copy()
    for step in [0.0, 0.2, 0.2, 2.0, 0.2, 6.0, 0.2]:
        x = x + step * 0.05 * rng.uniform(-1.0, 1.0, len(x)) / np.sqrt(2)
        want_value, want_gradient = objective(x)
        value, gradient = core_objective(x)
        diffs["objective"].add(value, want_value)
        diffs["objective"].add(gradient, want_gradient)
    if REPORT:
        with open(REPORT, "a") as out:
            for k, d in diffs.items():
                out.write(
                    f"{name}\t{k}\t{d.values}\t{d.bitwise}\t{d.abs:.3e}\t{d.rel:.3e}\n"
                )
    return diffs


def check_free(
    problem: Problem,
    core: _core.Problem,
    polylines: list[FloatArray],
    positions: list[FloatArray],
    diff: Diff,
) -> None:
    count = len(problem.members)
    at = _positions(problem, positions)
    copper = [[_field(via.given.copper)] for via in problem.vias]
    stops = [
        [None if f is None else _field(f) for f in via.given.stop_parts]
        for via in problem.vias
    ]
    for parts, sight, start, measured in (
        (copper, problem.via_sight, problem.inside, problem.via_distances),
        (stops, problem.via_stop, problem.stopped, problem.stop_distances),
    ):
        sight_core = core_sight(sight, count)
        for below in (-INSIDE_TOLERANCE, math.inf):
            want = measured(polylines, positions, below)
            got = _core.free_distances(parts, sight_core, polylines, at, below)
            assert got.static.key.tolist() == want.static.key.tolist()
            assert got.static.via.tolist() == want.static.via.tolist()
            diff.add(got.static.distance, want.static.distance)
            for name in ("key", "via", "obstacle", "vertex"):
                assert (
                    getattr(got.by_obstacle, name).tolist()
                    == getattr(want.by_obstacle, name).tolist()
                ), name
            diff.add(got.by_obstacle.distance, want.by_obstacle.distance)
            if below < 0:
                start_core = _core.free_distances(
                    parts,
                    sight_core,
                    problem.points,
                    _positions(problem, problem.positions),
                    below,
                )
                inside = newly_inside(want, start, INSIDE_TOLERANCE, DEEPER_ROUND_OFF)
                core_inside = _core.newly_inside(
                    got, start_core, INSIDE_TOLERANCE, DEEPER_ROUND_OFF
                )
                assert core_inside[0].tolist() == inside[0].tolist()
                assert core_inside[1].tolist() == inside[1].tolist()


def problems_of(
    chains: Sequence[ChainInput],
    pairs: Sequence[PairInput] = (),
    vias: Sequence[ViaInput] = (),
    params: EnergyParams = scenarios.PARAMS,
    static: Sequence[StaticVia] = (),
) -> list[Problem]:
    blocks = [pair_block(pair, DEFAULT_SETTINGS.spacing) for pair in pairs]
    everything = [*chains, *(b.chain for b in blocks)]
    moving = MovingVias(everything, vias, HOLE_CLEARANCE, HOLE_TO_HOLE)
    return [
        p.problem
        for p in optimize_module._problems(  # pyright: ignore[reportPrivateUsage]
            list(chains), blocks, moving, params, DEFAULT_SETTINGS, static
        )
    ]


def scenario_problems(make: Callable[[], AnyScenario]) -> list[Problem]:
    s = make()
    params = replace(s.params, repulsion=0.5, falloff_range=1.0)
    if isinstance(s, Scenario):
        alone = ChainInput(
            ChainCentreline("chain", "", 0.0, 0.0, s.polyline), "", s.fixed, s.field
        )
        return problems_of([alone], params=s.params)
    pairs = s.pairs if isinstance(s, PairsScenario) else ()
    vias = s.vias if isinstance(s, ViasScenario) else ()
    return problems_of(list(s.chains), pairs, vias, params)


SINGLE = [
    scenarios.wrap,
    lambda: scenarios.channel(1.0),
    scenarios.topology,
    scenarios.pad_row,
    scenarios.starts_inside,
    scenarios.detour,
    scenarios.narrow_gap,
]
MORE: list[Callable[[], AnyScenario]] = [
    *SINGLE,
    scenarios.tent,
    lambda: scenarios.via_slack(FreeSpace()),
    lambda: scenarios.via_over_bowed("N"),
    lambda: scenarios.via_over_bowed("P"),
    lambda: scenarios.via_detour_over(FreeSpace()),
    lambda: scenarios.via_grid(3, 4),
    scenarios.pair_tent_under_roof,
    *test_shared_search.SCENARIOS,
    *test_optimize_vias.VIA_ENERGY_SCENARIOS.values(),
]


@pytest.mark.parametrize("k", range(len(MORE)))
def test_scenarios(k: int) -> None:
    for n, problem in enumerate(scenario_problems(MORE[k])):
        check_problem(problem, f"scenario-{k}-{n}")


def test_contacts() -> None:
    chains = [
        test_optimize_contacts.end_in_pad(),
        test_optimize_contacts.passing_over(),
    ]
    for n, problem in enumerate(problems_of(chains)):
        check_problem(problem, f"contacts-{n}")


BUS_PITCH = 0.6


def _rows(count: int, jitter: float) -> list[float]:
    off = np.random.default_rng(1).uniform(-jitter, jitter, count)
    return [i * BUS_PITCH + float(o) for i, o in enumerate(off)]


def z_bus(jitter: float, rows: list[float] | None = None) -> list[ChainInput]:
    return [
        scenarios.selected(
            f"S{i}",
            f"S{i}",
            FreeSpace(),
            (0.0, y),
            (10.0 - y, y),
            (16.0 - y, y + 6.0),
            (30.0, y + 6.0),
        )
        for i, y in enumerate(rows or _rows(10, jitter))
    ]


def short_chains() -> list[ChainInput]:
    return [
        scenarios.selected(
            f"S{i}", f"S{i}", FreeSpace(), (0.0, y), (2.5, y + 0.15), (5.0, y)
        )
        for i, y in enumerate(_rows(262, 0.03))
    ]


def lone() -> ChainInput:
    y = -50.0
    return scenarios.selected(
        "lone",
        "lone",
        FreeSpace(),
        (0.0, y),
        (10.0, y),
        (16.0, y + 6.0),
        (30.0, y + 6.0),
    )


BUS_PARAMS = replace(scenarios.PARAMS, repulsion=0.5)


@pytest.mark.parametrize(
    "name",
    ["z", "z-jitter", "short", "one-net", "joint-z", "joint-pressed"],
)
def test_bus_and_joint(name: str) -> None:
    pressed = scenarios.PAIR_WIDTH + scenarios.PAIR_CLEARANCE
    chains = {
        "z": lambda: z_bus(0.0),
        "z-jitter": lambda: z_bus(0.03),
        "short": short_chains,
        "one-net": lambda: [
            replace(c, centreline=replace(c.centreline, net="GND"))
            for c in short_chains()
        ],
        "joint-z": lambda: [lone(), *z_bus(0.0)],
        "joint-pressed": lambda: [
            lone(),
            *z_bus(0.0, [i * pressed for i in range(10)]),
        ],
    }[name]()
    for n, problem in enumerate(problems_of(chains, params=BUS_PARAMS)):
        check_problem(problem, f"bus-{name}-{n}")


def test_crossed_grid() -> None:
    pitch, count = 1.0, 5
    size = (count - 1) * pitch
    region = box(-1.0, -1.0, size + 1.0, size + 1.0)
    zone = ViaZone("GND", {layer: region for layer in ("F.Cu", "B.Cu")}, 0.2, 0.0)
    at = (2.5 * pitch, 1.5 * pitch)
    signal = scenarios.through_via("x0", "S0", at)
    chains = scenarios.route("S0", at, (-3.0, at[1] - 1.5), (size + 3.0, at[1] + 1.5))
    vias = [scenarios.joining(signal, 0)]
    for i in range(count):
        for j in range(count):
            p = (i * pitch, j * pitch)
            if math.dist(p, at) > 0.85:
                via = scenarios.through_via(f"g{i}_{j}", "GND", p)
                vias.append(
                    ViaInput(
                        via,
                        (),
                        FreeSpace(),
                        scenarios.PAIR_CLEARANCE,
                        zones=(zone,),
                        ground=True,
                    )
                )
    params = replace(scenarios.PARAMS, attraction=0.25)
    for n, problem in enumerate(problems_of(chains, vias=vias, params=params)):
        check_problem(problem, f"crossed-{n}")


class Captured(Exception):
    pass


def _pipeline(
    board: CommitBoard, args: list[str], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def kicad(**_: object) -> FakeKiCad:
        return FakeKiCad(board)

    monkeypatch.setattr(pipeline.kipy, "KiCad", kicad)
    real = optimize_module._problems  # pyright: ignore[reportPrivateUsage]

    def capturing(
        chains: Sequence[ChainInput],
        blocks: Sequence[PairBlock],
        vias: MovingVias,
        params: EnergyParams,
        settings: OptimizerSettings,
        static_vias: Sequence[StaticVia] = (),
    ) -> list[Placed]:
        placed = real(chains, blocks, vias, params, settings, static_vias)
        for n, p in enumerate(placed):
            check_problem(p.problem, f"{name}-{n}")
        raise Captured

    monkeypatch.setattr(optimize_module, "_problems", capturing)
    options = cli.smooth_options(cli.parse_args(["smooth", "--dry-run", *args]))
    with pytest.raises(Captured):
        pipeline.run_smooth(options, lambda _: None)


@pytest.mark.parametrize(
    "case",
    [((5.0, 1.5), False), ((6.06, 1.06), False), ((5.0, 2.5), True)],
)
@pytest.mark.parametrize("args", [[], ["--repulsion", "0"]])
def test_return_vias(
    case: tuple[tuple[float, float], bool],
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ground, stitching = case
    _pipeline(
        return_via_board(ground, stitching=stitching),
        args,
        monkeypatch,
        f"return-{ground}-{stitching}-{args}",
    )


def test_pad_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import itertools

    tracks: list[Track] = []
    for i in range(10):
        net = f"S{i}"
        y = i * 2.0
        j = y + 0.3
        corners = [(0.0, y), (10.0, y), (10.6, j), (20.0, j), (20.6, y), (30.0, y)]
        tracks += [
            kipy_track(f"{net}_{k}", net, s, e)
            for k, (s, e) in enumerate(itertools.pairwise(corners))
        ]
    pads = [
        kipy_pad(f"p{row}_{k}", f"P{row}_{k}", k * 1.0, row * 2.0 + 1.15)
        for row in range(-1, 10)
        for k in range(31)
    ]
    board = CommitBoard(selection=list(tracks), tracks=list(tracks), pads=pads)
    _pipeline(board, [], monkeypatch, "pad-rows")
