"""Measure the optimizer: wall and CPU time, the seconds of setup and those
after it outside L-BFGS-B, outer iterations, bursts, energy evaluations, the
times a layer's neighbour list found its pairs (`seen.NeighbourList`), both
from the Rust statistics (`kibends._core.statistics`) or, for a tree without
them, by patching its Python search, and
the farthest move in one burst on stitched ground planes, on the synthetic
scenarios and on buses of parallel tracks, how a lone track settles in one
problem with a cluster of others, and where return vias settle.

Usage: ./run bench [--tree DIR]
                   [--only grid|crossed|scenarios|bus|return|joint]
                   [--case TEXT] [--plot DIR]
                   [-- SMOOTH_ARGS]

`--tree` measures another checkout's source and tests, such as an older
commit unpacked with `git archive`, so that two versions can be compared
on one machine. `--case` runs only the planes, grids, buses and joint
cases whose names contain TEXT. `--plot` saves each joint case's
per-chain length against outer iteration as a PNG in DIR.
Arguments after `--` go to `kibends smooth` for the runs through a fake
KiCad and give the crossed grids' energy parameters and optimizer settings
and the buses' optimizer settings, a `--time-limit` among them overriding
TIME_LIMIT, so a run can be measured at other settings. Each table row is
one run.

Setup ends when the optimizer logs that its problems are built; in a run
through a fake KiCad it holds reading the board and building the obstacle
fields too. The seconds outside L-BFGS-B are the rest of the run but
L-BFGS-B and the energy evaluations it makes: resampling, the checks
around each burst, the guard and, through a fake KiCad, simplifying.

- grid: two-layer GND planes whose stitching vias are all selected, among
  them a 30 mm plane with 194 vias at 1.72 mm pitch, with and without three
  signal routes that change layers through a via inside it, run end to end
  through a fake KiCad as `kibends smooth --dry-run` runs, setup included,
  each stopped after TIME_LIMIT seconds unless it converges first.
- crossed: GND stitching grids in a GND zone that signal routes cross,
  each through the optimizer at default settings, stopped after TIME_LIMIT
  seconds unless it converges first.
- scenarios: the cases of `tests/scenarios.py`, each through the optimizer
  at default settings; width-change and starts-inside are each skipped for
  a tree without them (an older tree's neck-down is another geometry).
- bus: tracks of different nets side by side, each case stopped after
  TIME_LIMIT seconds unless it ends first. Five go through the optimizer
  in free space with the scenarios' energy parameters and a repulsion of
  0.5: ten Z-shaped tracks at 0.6 mm pitch, each as far from one neighbour
  as from the other; the same with each track up to 0.03 mm off its pitch;
  262 chains of 5 mm for one outer iteration, to time setup and the
  work around a burst among many chains; the same, then simplified, the
  simplification timed in its note and counted in its wall time and its
  time outside L-BFGS-B; and the same chains all of one
  net with one of another net far from them. The last runs
  end to end
  through a fake KiCad as the planes do: ten tracks 2 mm apart between
  rows of pads at 1 mm pitch, whose distance field every energy evaluation
  queries.
- joint: a lone free track, run alone and in one problem with a cluster of
  chains pressed against each other at clearance far from it on its layer,
  so that the two cannot interact. Each burst records every chain's length
  and its shape change (`polylines.shape_change`); the table gives the lone
  track's length when the alone run settled and when the joint run did.
- return: a signal via and a ground via settling together, at default
  settings and without repulsion; skipped for a tree without return-via
  attraction or without the test helpers it builds its boards with.
"""

import argparse
import importlib
import itertools
import math
import sys
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the optimizer.")
    parser.add_argument("--tree", type=Path, default=ROOT)
    parser.add_argument(
        "--only", choices=("grid", "crossed", "scenarios", "bus", "return", "joint")
    )
    parser.add_argument("--case", default="", help="run only the cases naming this")
    parser.add_argument(
        "--plot", type=Path, help="directory for the joint cases' length plots"
    )
    parser.add_argument("smooth", nargs="*", help="arguments for kibends smooth")
    return parser.parse_args()


ARGS = parse_args()
TREE = ARGS.tree.resolve()
sys.path[:0] = [str(TREE / "tests"), str(TREE / "src")]

# Imported from the tree measured, after its paths are set.
import conftest  # noqa: E402
import numpy as np  # noqa: E402
import scenarios  # noqa: E402
from conftest import (  # noqa: E402
    CommitBoard,
    FakeKiCad,
    kipy_copper_zone,
    kipy_pad,
    kipy_rect,
    kipy_track,
    kipy_via,
)
from distance_fields import Box, FreeSpace  # noqa: E402
from kipy.board_types import Track, Via  # noqa: E402
from kipy.proto.board.board_types_pb2 import BoardLayer  # noqa: E402
from loguru import logger  # noqa: E402
from scenarios import ChainsScenario, PairsScenario, ViasScenario  # noqa: E402
from shapely.geometry import box  # noqa: E402

import kibends  # noqa: E402
from kibends import bursts as bursts_module  # noqa: E402
from kibends import cli  # noqa: E402
from kibends import optimize as optimize_module  # noqa: E402
from kibends import run as pipeline  # noqa: E402
from kibends.bursts import (  # noqa: E402
    Burst,
    Objective,
    OptimizerSettings,
    Revert,
)
from kibends.chain import NM_PER_MM  # noqa: E402
from kibends.energy import EnergyParams, FloatArray  # noqa: E402
from kibends.neighbours import ChainInput  # noqa: E402
from kibends.optimize import (  # noqa: E402
    OptimizeResult,
    RunResult,
    optimize,
    optimize_chains,
)
from kibends.polylines import shape_change  # noqa: E402
from kibends.simplify import simplify_chains  # noqa: E402
from kibends.snapshot import ViaZone  # noqa: E402
from kibends.vias import ViaInput  # noqa: E402

assert Path(kibends.__file__).is_relative_to(TREE), kibends.__file__


SET_UP = "Optimizer set up"
"""How the line starts that the optimizer logs once its problems are built."""


@dataclass
class Count:
    bursts: int = 0
    evaluations: int = 0
    searches: int = 0
    """How many times a layer's neighbour list found its pairs."""
    step: float = 0.0
    """The farthest any vertex or via centre moved in one burst, mm."""
    ready: float | None = None
    """When the optimizer first had its problems built, as
    `time.perf_counter` reads; None while it has not."""
    minimising: float = 0.0
    """Seconds L-BFGS-B ran, its energy evaluations included."""


Statistics = Callable[[], tuple[int, int]]
"""The energy evaluations and neighbour-list searches so far."""


def python_search() -> bool:
    """Whether the measured tree's neighbour lists search in Python
    (`seen.Seen.candidates`), as before its energy moved to Rust."""
    seen = getattr(importlib.import_module("kibends.seen"), "Seen")  # noqa: B009
    return hasattr(seen, "candidates")


def rust_statistics() -> Statistics | None:
    """The measured tree's Rust statistics, None for a tree that searches
    in Python."""
    if python_search():
        return None
    from kibends._core import statistics

    def read() -> tuple[int, int]:
        counts = statistics()
        return counts.evaluations, counts.list_searches

    return read


def count_python_searches(count: Count) -> Callable[[], None]:
    """Count a tree's Python neighbour-list searches (`seen.Seen.candidates`)
    in `count`; returns what undoes it."""
    # Looked up by name: this tree's own `seen` has no Python search.
    module = importlib.import_module("kibends.seen")
    seen: type = getattr(module, "Seen")  # noqa: B009
    candidates: Callable[..., object] = getattr(seen, "candidates")  # noqa: B009

    def searched(*args: object) -> object:
        count.searches += 1
        return candidates(*args)

    setattr(seen, "candidates", searched)  # noqa: B010
    return lambda: setattr(seen, "candidates", candidates)


@contextmanager
def counted() -> Generator[Count]:
    """Count the bursts the optimizer runs and the energy evaluations in
    them, and time its setup and L-BFGS-B."""
    count = Count()
    real = optimize_module.run_burst
    statistics = rust_statistics()
    started = None if statistics is None else statistics()

    def heard(message: str) -> None:
        if count.ready is None and message.startswith(SET_UP):
            count.ready = time.perf_counter()

    def burst(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Callable[[], bool] | None = None,
    ) -> Burst:
        count.bursts += 1
        checking = 0.0

        def counting(x: FloatArray) -> tuple[float, FloatArray]:
            count.evaluations += 1
            return objective(x)

        # A tree with Rust statistics counts its evaluations there, and
        # this tree's bursts evaluate only its Rust objective. An older
        # tree's is a Python function, counted here; a scaled one keeps its
        # scale, and a still older tree has none.
        evaluated = objective
        if statistics is None:
            evaluated = cast(Objective, counting)
            if hasattr(bursts_module, "ScaledObjective") and isinstance(
                objective, bursts_module.ScaledObjective
            ):
                scaled = cast(Callable[..., Objective], bursts_module.ScaledObjective)
                evaluated = scaled(counting, objective.scale)

        def checked(x: FloatArray) -> int:
            nonlocal checking
            began = time.perf_counter()
            crossed = revert(x)
            checking += time.perf_counter() - began
            return crossed

        began = time.perf_counter()
        # An older tree's bursts take no stop.
        result = (
            real(evaluated, start, checked, settings)
            if stop is None
            else real(evaluated, start, checked, settings, stop)
        )
        count.minimising += time.perf_counter() - began - checking
        if len(start):
            moved = (result.points - start).reshape(-1, 2)
            count.step = max(count.step, float(np.linalg.norm(moved, axis=1).max()))
        return result

    setattr(optimize_module, "run_burst", burst)  # noqa: B010
    undo = None if statistics is not None else count_python_searches(count)
    sink = logger.add(heard, level="INFO", format="{message}")
    try:
        yield count
    finally:
        logger.remove(sink)
        if undo is not None:
            undo()
        if statistics is not None and started is not None:
            evaluations, searches = statistics()
            count.evaluations = evaluations - started[0]
            count.searches = searches - started[1]
        setattr(optimize_module, "run_burst", real)  # noqa: B010


@dataclass(frozen=True)
class Measured:
    wall: float
    cpu: float
    setup: float
    """Seconds until the optimizer had its problems built: the whole run if
    it never had."""
    outside: float
    """Seconds after setup outside L-BFGS-B."""
    iterations: int
    count: Count
    note: str


def measure(name: str, work: Callable[[], tuple[int, str]]) -> Measured:
    with counted() as count:
        began, cpu = time.perf_counter(), time.process_time()
        iterations, note = work()
        ended, cpu = time.perf_counter(), time.process_time() - cpu
    wall = ended - began
    setup = (ended if count.ready is None else count.ready) - began
    outside = wall - setup - count.minimising
    measured = Measured(wall, cpu, setup, outside, iterations, count, note)
    print(
        f"| {name} | {wall:.2f} | {cpu:.2f} | {setup:.2f} | {outside:.2f} "
        f"| {iterations} | {count.bursts} | {count.evaluations} "
        f"| {count.searches} | {count.step:.2f} | {note} |",
        flush=True,
    )
    return measured


HEADER = (
    "| case | wall s | CPU s | setup s | outside L-BFGS-B s | iterations | bursts "
    "| evaluations | list searches | largest step mm | result |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)

ROUTE_KEEPOUT = 0.7
"""How near a route's track no stitching via is placed, mm."""
ROUTE_ROWS = (5.3 / 30, 12.7 / 30, 20.1 / 30)
"""Where the signal routes start, as fractions of the plane's size."""
ROUTE_COLUMNS = (2.1 / 30, 15.1 / 30, 28.1 / 30)
"""Where a route starts, drops to its via and ends, as fractions of the
plane's size."""
ROUTE_DROP = 0.2
"""How far a route runs down to its via, as a fraction of the plane's size."""
FAR = -5.0
"""The height of the track selected with no signal route, mm."""
TIME_LIMIT = 25.0
"""Seconds after which a plane run stops, keeping each measurement under
30 s; a later `--time-limit` overrides it."""
B_CU = BoardLayer.BL_B_Cu


@dataclass(frozen=True)
class Plane:
    """A square two-layer GND plane of `size` mm with GND stitching vias on
    a grid of `pitch` mm, `count` to a side but those on a route's copper,
    and `routes` signal routes: each along F.Cu into the plane, down to a
    via and on along B.Cu, with a corner on each side to cut. A plane
    without routes keeps the gaps three routes leave in its grid."""

    count: int
    pitch: float
    routes: int
    size: float

    def tracks(self, routes: int) -> tuple[list[Track], list[Via]]:
        tracks: list[Track] = []
        vias: list[Via] = []
        s = self.size
        rows = ROUTE_ROWS[1:2] if routes == 1 else ROUTE_ROWS[:routes]
        for k, row in enumerate(rows):
            net = f"SIG{k}"
            y, low = row * s, (row + ROUTE_DROP) * s
            start, middle, end = (c * s for c in ROUTE_COLUMNS)
            tracks += [
                kipy_track(f"{net}a", net, (start, y), (middle, y)),
                kipy_track(f"{net}b", net, (middle, y), (middle, low)),
                kipy_track(f"{net}c", net, (middle, low), (end, low), layer=B_CU),
                kipy_track(f"{net}d", net, (end, low), (end, y), layer=B_CU),
            ]
            vias.append(kipy_via(f"x{k}", net, middle, low))
        return tracks, vias

    def board(self) -> CommitBoard:
        """The plane with its stitching vias and routes all selected."""
        tracks, signal_vias = self.tracks(self.routes)
        keepout = self.tracks(self.routes or len(ROUTE_ROWS))[0]
        stitching = [
            kipy_via(f"g{i}_{j}", "GND", i * self.pitch, j * self.pitch)
            for i in range(1, self.count + 1)
            for j in range(1, self.count + 1)
            if clear_of(i * self.pitch, j * self.pitch, keepout)
        ]
        if not tracks:
            # The selection needs a track: a straight one, out of every via's reach.
            tracks = [kipy_track("far", "FAR", (0.0, FAR), (self.size, FAR))]
        outline = kipy_rect(-1, -1, self.size + 1, self.size + 1)
        return CommitBoard(
            selection=[*tracks, *stitching, *signal_vias],
            tracks=list(tracks),
            vias=[*stitching, *signal_vias],
            zones=[kipy_copper_zone("gnd", "GND", outline, [BoardLayer.BL_F_Cu, B_CU])],
        )

    def on_edge(self, via_id: str) -> bool:
        """Whether a via is a stitching via on the grid's outer row."""
        if not via_id.startswith("g"):
            return False
        return any(int(k) in (1, self.count) for k in via_id[1:].split("_"))


PLANES = {
    "194 vias at 1.72 mm, 3 routes": Plane(16, 1.72, 3, 30.0),
    "194 vias at 1.72 mm, no route": Plane(16, 1.72, 0, 30.0),
    "9x9 at 1.27 mm, 3 routes": Plane(9, 1.27, 3, 12.7),
    "6x6 at 1.27 mm, 1 route": Plane(6, 1.27, 1, 8.9),
    "7x7 at 1.5 mm, 1 route": Plane(7, 1.5, 1, 12.0),
}


def clear_of(x: float, y: float, tracks: Sequence[Track]) -> bool:
    for t in tracks:
        (x0, y0), (x1, y1) = sorted(
            (
                (t.start.x / NM_PER_MM, t.start.y / NM_PER_MM),
                (t.end.x / NM_PER_MM, t.end.y / NM_PER_MM),
            )
        )
        if x0 - ROUTE_KEEPOUT < x < x1 + ROUTE_KEEPOUT and (
            y0 - ROUTE_KEEPOUT < y < y1 + ROUTE_KEEPOUT
        ):
            return False
    return True


def connect(board: CommitBoard) -> None:
    """Point the pipeline's KiCad connection at `board`."""

    def kicad(**_: object) -> FakeKiCad:
        return FakeKiCad(board)

    setattr(pipeline.kipy, "KiCad", kicad)  # noqa: B010


def smooth_board(
    board: CommitBoard, options: pipeline.SmoothOptions
) -> pipeline.OptimizerReport:
    """Run `kibends smooth` with `options` on `board` through a fake KiCad;
    returns the optimizer's report."""
    connect(board)
    reports: list[pipeline.OptimizerReport] = []
    pipeline.run_smooth(options, reports.append)
    (report,) = reports
    return report


def plane_run(plane: Plane) -> Callable[[], tuple[int, str]]:
    board = plane.board()
    options = replace(smooth_options(), dry_run=True)

    def go() -> tuple[int, str]:
        report = smooth_board(board, options)
        moves = {v.id: math.dist(v.start, v.end) for v in report.vias}
        rows = {
            "outer row": [m for k, m in moves.items() if plane.on_edge(k)],
            "inside": [
                m
                for k, m in moves.items()
                if k.startswith("g") and not plane.on_edge(k)
            ],
            "signal vias": [m for k, m in moves.items() if not k.startswith("g")],
        }
        farthest = ", ".join(f"{max(m):.2f} {row}" for row, m in rows.items() if m)
        length = sum(c.after.length for c in report.chains if c.net != "FAR")
        return report.iterations, (
            f"{report.termination.value}; {sum(m > 0 for m in moves.values())} "
            f"of {len(board.vias)} vias moved, farthest (mm): {farthest}; "
            f"chains {length:.3f} mm"
        )

    return go


def grid() -> None:
    print(HEADER)
    for name, plane in PLANES.items():
        if ARGS.case in name:
            measure(name, plane_run(plane))


Case = Callable[[], tuple[int, str]]
"""A run to measure, which returns its outer iterations and outcome."""
Several = ChainsScenario | ViasScenario | PairsScenario


def scenario_makers() -> dict[str, Callable[[], scenarios.Scenario | Several]]:
    """The scenarios of `tests/scenarios.py` the benchmark runs, by name."""
    free = FreeSpace()
    return {
        "wrap": scenarios.wrap,
        "channel-0": lambda: scenarios.channel(0.0),
        "channel-1": lambda: scenarios.channel(1.0),
        "topology": scenarios.topology,
        "pad-row": scenarios.pad_row,
        **(
            {"starts-inside": scenarios.starts_inside}
            if hasattr(scenarios, "starts_inside")
            else {}
        ),
        "detour": scenarios.detour,
        "narrow-gap": scenarios.narrow_gap,
        "channel-pair": scenarios.channel_pair,
        "uneven-channel-pair": scenarios.uneven_channel_pair,
        "tent": scenarios.tent,
        "same-net-wrap": scenarios.same_net_wrap,
        "crowded": scenarios.crowded,
        "drape": scenarios.drape,
        "squeeze": scenarios.squeeze,
        "hooks": scenarios.hooks,
        # An older tree's neck-down holds its corner, so it is no stand-in.
        **(
            {"width-change": scenarios.width_change}
            if hasattr(scenarios, "width_change")
            else {}
        ),
        "centreline-over-pad": scenarios.centreline_over_pad,
        "via-detour": scenarios.via_detour,
        "via-on-one-layer": scenarios.via_on_one_layer,
        "via-fermat": scenarios.via_fermat,
        "via-hemmed": scenarios.via_hemmed,
        "via-corner": scenarios.via_corner,
        "via-beside-chain": scenarios.via_beside_chain,
        "via-pair": scenarios.via_pair,
        "via-over-pinned-N": lambda: scenarios.via_over_pinned("N"),
        "via-over-pinned-P": lambda: scenarios.via_over_pinned("P"),
        "via-over-bowed-N": lambda: scenarios.via_over_bowed("N"),
        "via-over-bowed-P": lambda: scenarios.via_over_bowed("P"),
        "via-detour-over": lambda: scenarios.via_detour_over(free),
        "via-detour-over-box": lambda: scenarios.via_detour_over(
            Box(4.9, 2.0, 0.05, 0.05)
        ),
        "via-slack": lambda: scenarios.via_slack(free),
        "via-across-chain": scenarios.via_across_chain,
        "via-grid-3x4": lambda: scenarios.via_grid(3, 4),
        "pair-over-pad": scenarios.pair_over_pad,
        "pair-tent": scenarios.pair_tent,
        "pair-corner": scenarios.pair_corner,
        "pair-round-corner": scenarios.pair_round_corner,
        "pair-uneven-end": scenarios.pair_uneven_end,
        "pair-chamfered": scenarios.pair_chamfered,
        "pair-split": scenarios.pair_split,
        "pair-turning-back-150": scenarios.pair_turning_back,
    }


def run_scenario(
    s: scenarios.Scenario | Several,
) -> OptimizeResult | RunResult:
    """The scenario through the optimizer at default settings."""
    if isinstance(s, scenarios.Scenario):
        return optimize(s.polyline, s.fixed, s.field, s.params)
    return optimize_chains(
        s.chains,
        s.params,
        vias=s.vias if isinstance(s, ViasScenario) else (),
        pairs=s.pairs if isinstance(s, PairsScenario) else (),
    )


def scenario_cases() -> dict[str, Case]:
    def case(make: Callable[[], scenarios.Scenario | Several]) -> Case:
        def go() -> tuple[int, str]:
            result = run_scenario(make())
            if isinstance(result, OptimizeResult):
                total = length(result.polyline)
            else:
                total = sum(length(c.polyline) for c in result.chains) + sum(
                    length(t) for p in result.pairs for t in p.kept()
                )
            return result.iterations, f"{result.termination.value}, {total:.4f} mm"

        return go

    return {name: case(make) for name, make in scenario_makers().items()}


def length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def scenarios_table() -> None:
    if not hasattr(scenarios, "width_change"):
        print("scenarios: width-change skipped, the tree has no width_change")
    print(HEADER)
    total = Count()
    wall = cpu = setup = outside = 0.0
    iterations = 0
    for name, case in scenario_cases().items():
        m = measure(name, case)
        wall, cpu, iterations = wall + m.wall, cpu + m.cpu, iterations + m.iterations
        setup, outside = setup + m.setup, outside + m.outside
        total.bursts += m.count.bursts
        total.evaluations += m.count.evaluations
        total.searches += m.count.searches
        total.step = max(total.step, m.count.step)
    print(
        f"| **all {len(scenario_cases())}** | {wall:.2f} | {cpu:.2f} | {setup:.2f} "
        f"| {outside:.2f} | {iterations} | {total.bursts} | {total.evaluations} "
        f"| {total.searches} | {total.step:.2f} | |"
    )
    if not hasattr(scenarios, "starts_inside"):
        print("starts-inside: skipped, the tree has no scenarios.starts_inside")


CROSSED = {
    "crossed 9x9 at 1.27 mm, 3 routes": (9, 1.27, 3),
    "crossed 6x6 at 1.27 mm, 1 route": (6, 1.27, 1),
    "crossed 7x7 at 1.5 mm, 1 route": (7, 1.5, 1),
}
"""Stitching grids, by vias to a side, pitch in mm and signal routes."""
CROSSED_SPAN = 3.0
"""How far a crossing route starts and ends beyond the grid, mm."""
CROSSED_RISE = 1.5
"""How far a crossing route rises to each side of its via, mm."""
SIGNAL_KEEPOUT = 0.85
"""How near a signal via no stitching via is placed, mm: farther than its
ring, the stitching via's and the clearance."""


def crossed_grid(count: int, pitch: float, routes: int) -> ViasScenario:
    """GND stitching vias on a square grid in a GND zone on F.Cu and B.Cu,
    and signal routes crossing it, F.Cu to a via amid the grid and B.Cu on,
    each rising across it."""
    size = (count - 1) * pitch
    region = box(-1.0, -1.0, size + 1.0, size + 1.0)
    zone = ViaZone("GND", {layer: region for layer in ("F.Cu", "B.Cu")}, 0.2, 0.0)
    params = smooth_options().params
    chains: list[ChainInput] = []
    vias: list[ViaInput] = []
    for k in range(routes):
        row = round(size * (k + 1) / (routes + 1) / pitch - 0.5) + 0.5
        at = ((round(count / 2) - 0.5) * pitch, row * pitch)
        start = (-CROSSED_SPAN, at[1] - CROSSED_RISE)
        end = (size + CROSSED_SPAN, at[1] + CROSSED_RISE)
        net = f"S{k}"
        signal = scenarios.through_via(f"x{k}", net, at)
        vias.append(scenarios.joining(signal, len(chains)))
        chains += scenarios.route(net, at, start, end)
    signals = [v.via.position for v in vias]
    for i in range(count):
        for j in range(count):
            at = (i * pitch, j * pitch)
            if all(math.dist(at, s) > SIGNAL_KEEPOUT for s in signals):
                via = scenarios.through_via(f"g{i}_{j}", "GND", at)
                stitching = ViaInput(
                    via, (), FreeSpace(), scenarios.PAIR_CLEARANCE, zones=(zone,)
                )
                if getattr(params, "attraction", 0.0):
                    stitching = replace(stitching, ground=True)
                vias.append(stitching)
    return ViasScenario("crossed", tuple(chains), tuple(vias), params)


def smooth_options() -> pipeline.SmoothOptions:
    """The options `kibends smooth` takes with the arguments given after
    `--`, its time limit TIME_LIMIT unless they give one."""
    args = ["smooth", f"--time-limit={TIME_LIMIT:g}", *ARGS.smooth]
    return cli.smooth_options(cli.parse_args(args))


def crossed_run(scenario: ViasScenario) -> Case:
    settings = smooth_options().settings

    def go() -> tuple[int, str]:
        result = optimize_chains(
            scenario.chains, scenario.params, settings, scenario.vias
        )
        total = sum(length(c.polyline) for c in result.chains)
        moved = max(
            math.dist(v.position, given.via.position)
            for v, given in zip(result.vias, scenario.vias, strict=True)
            if given.via.net == "GND"
        )
        return result.iterations, (
            f"{result.termination.value}; {len(scenario.vias)} vias, stitching "
            f"moved up to {moved:.2f} mm; chains {total:.3f} mm"
        )

    return go


def crossed() -> None:
    print(HEADER)
    for name, (count, pitch, routes) in CROSSED.items():
        if ARGS.case in name:
            measure(name, crossed_run(crossed_grid(count, pitch, routes)))


BUS_TRACKS = 10
BUS_PITCH = 0.6
"""How far apart a bus's tracks lie, mm."""
BUS_JITTER = 0.03
"""How far off its pitch an unevenly placed track lies at most, mm."""
BUS_SEED = 1
"""Seeds the uneven tracks' offsets, so that every run places them alike."""
BUS_REPULSION = 0.5
SHORT_CHAINS = 262
FAR_TRACK = 50.0
"""How far below the chains of one net the chain of another lies, mm."""
PAD_TRACK_LENGTH = 30.0
"""How far the tracks between the pad rows run, mm."""
PAD_TRACK_PITCH = 2.0
"""How far apart they lie, mm."""
PAD_TRACK_JOG = 0.3
"""How far each of them steps aside over its middle third, mm."""
PAD_PITCH = 1.0
PAD_ROW_OFFSET = 1.15
"""How far a row of pads lies from the track before it, mm."""


def bus_rows(count: int, jitter: float) -> list[float]:
    """Where `count` tracks BUS_PITCH apart lie, each up to `jitter` mm off
    its place, the same in every run."""
    off = np.random.default_rng(BUS_SEED).uniform(-jitter, jitter, count)
    return [i * BUS_PITCH + float(o) for i, o in enumerate(off)]


def z_bus(jitter: float) -> list[ChainInput]:
    """BUS_TRACKS tracks of as many nets, each 30 mm across with a diagonal
    6 mm down in between, nested, in free space."""
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
        for i, y in enumerate(bus_rows(BUS_TRACKS, jitter))
    ]


def short_chains() -> list[ChainInput]:
    """SHORT_CHAINS chains of as many nets, each 5 mm across with a 0.15 mm
    bump, unevenly placed, in free space."""
    return [
        scenarios.selected(
            f"S{i}", f"S{i}", FreeSpace(), (0.0, y), (2.5, y + 0.15), (5.0, y)
        )
        for i, y in enumerate(bus_rows(SHORT_CHAINS, BUS_JITTER))
    ]


def one_net_chains() -> list[ChainInput]:
    """`short_chains`, all of one net, and one chain of another net
    FAR_TRACK below them."""
    chains = [
        replace(c, centreline=replace(c.centreline, net="GND")) for c in short_chains()
    ]
    far = scenarios.selected(
        "far",
        "S",
        FreeSpace(),
        (0.0, -FAR_TRACK),
        (2.5, 0.15 - FAR_TRACK),
        (5.0, -FAR_TRACK),
    )
    return [*chains, far]


def pad_rows_board() -> CommitBoard:
    """BUS_TRACKS selected tracks of as many nets, each with a jog over its
    middle third, and rows of pads, each pad of its own net: one row between
    every two tracks and one beyond each outer track."""
    tracks: list[Track] = []
    for i in range(BUS_TRACKS):
        net = f"S{i}"
        y = i * PAD_TRACK_PITCH
        jogged = y + PAD_TRACK_JOG
        corners = [
            (0.0, y),
            (10.0, y),
            (10.6, jogged),
            (20.0, jogged),
            (20.6, y),
            (PAD_TRACK_LENGTH, y),
        ]
        tracks += [
            kipy_track(f"{net}_{k}", net, start, end)
            for k, (start, end) in enumerate(itertools.pairwise(corners))
        ]
    pads = [
        kipy_pad(
            f"p{row}_{k}",
            f"P{row}_{k}",
            k * PAD_PITCH,
            row * PAD_TRACK_PITCH + PAD_ROW_OFFSET,
        )
        for row in range(-1, BUS_TRACKS)
        for k in range(round(PAD_TRACK_LENGTH / PAD_PITCH) + 1)
    ]
    return CommitBoard(selection=list(tracks), tracks=list(tracks), pads=pads)


def bus_note(
    termination: str, before: float, after: float, abnormal: Sequence[tuple[str, int]]
) -> str:
    """A bus run's outcome, its chains' summed length after and before, mm,
    and how many of its bursts L-BFGS-B ended abnormally."""
    bursts = sum(n for _, n in abnormal)
    return (
        f"{termination}; chains {after:.3f} mm, from {before:.3f} mm; "
        f"{bursts} abnormal burst{'' if bursts == 1 else 's'}"
    )


def bus_run(
    chains: Sequence[ChainInput],
    settings: OptimizerSettings,
    *,
    simplified: bool = False,
) -> Case:
    """The chains through the optimizer, and, if `simplified`, through
    simplification after it, timed in the note."""
    params = replace(scenarios.PARAMS, repulsion=BUS_REPULSION)

    def go() -> tuple[int, str]:
        result = optimize_chains(chains, params, settings)
        note = bus_note(
            result.termination.value,
            sum(length(c.centreline.points) for c in chains),
            sum(length(c.polyline) for c in result.chains),
            result.abnormal,
        )
        if simplified:
            began = time.perf_counter()
            simplify_chains(chains, result)
            note += f"; simplified in {time.perf_counter() - began:.2f} s"
        return result.iterations, note

    return go


def pad_rows_run() -> Case:
    board = pad_rows_board()
    options = replace(smooth_options(), dry_run=True)

    def go() -> tuple[int, str]:
        report = smooth_board(board, options)
        return report.iterations, bus_note(
            report.termination.value,
            sum(c.before.length for c in report.chains),
            sum(c.after.length for c in report.chains),
            report.abnormal,
        )

    return go


def bus_cases() -> dict[str, Case]:
    settings = smooth_options().settings
    return {
        "10 Z tracks at 0.6 mm, equal pitch": bus_run(z_bus(0.0), settings),
        "10 Z tracks at 0.6 mm, pitch varied by 0.03 mm": bus_run(
            z_bus(BUS_JITTER), settings
        ),
        "262 chains of 5 mm, 1 iteration": bus_run(
            short_chains(), replace(settings, max_iterations=1)
        ),
        "262 chains of 5 mm, 1 iteration, simplified": bus_run(
            short_chains(), replace(settings, max_iterations=1), simplified=True
        ),
        "262 chains of one net and one far off, 1 iteration": bus_run(
            one_net_chains(), replace(settings, max_iterations=1)
        ),
        "10 tracks between pad rows at 1 mm": pad_rows_run(),
    }


def bus() -> None:
    print(HEADER)
    for name, case in bus_cases().items():
        if ARGS.case in name:
            measure(name, case)


LONE = "lone"
"""The lone track's description."""
LONE_OFFSET = 50.0
"""How far below the cluster the lone track lies, mm: beyond any reach."""
PRESSED_PITCH = scenarios.PAIR_WIDTH + scenarios.PAIR_CLEARANCE
"""Centre to centre at exactly the clearance, mm."""


@dataclass
class ChainTrace:
    """One chain's length and shape change after each burst of its problem,
    mm, and the run's seconds when the burst ended."""

    lengths: list[float]
    moves: list[float]
    seconds: list[float]


@contextmanager
def traced() -> Generator[dict[str, ChainTrace]]:
    """Record every free member's length and shape change after each burst."""
    problem_class = optimize_module._Problem  # pyright: ignore[reportPrivateUsage]
    real = problem_class.step
    traces: dict[str, ChainTrace] = {}
    began = time.perf_counter()

    def step(
        problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
        stop: Callable[[], bool] | None = None,
    ) -> float:
        before = list(problem.points)
        change = real(problem, stop)
        now = time.perf_counter() - began
        for i in problem.free:
            name = problem.bodies[i].centreline.description
            trace = traces.setdefault(name, ChainTrace([], [], []))
            trace.lengths.append(length(problem.points[i]))
            trace.moves.append(shape_change(before[i], problem.points[i]))
            trace.seconds.append(now)
        return change

    setattr(problem_class, "step", step)  # noqa: B010
    try:
        yield traces
    finally:
        setattr(problem_class, "step", real)  # noqa: B010


def lone_track() -> ChainInput:
    """A Z track 30 mm across with a diagonal 6 mm down, in free space."""
    y = -LONE_OFFSET
    return scenarios.selected(
        LONE,
        LONE,
        FreeSpace(),
        (0.0, y),
        (10.0, y),
        (16.0, y + 6.0),
        (30.0, y + 6.0),
    )


def pressed_bus(count: int) -> list[ChainInput]:
    """`count` nested Z tracks as `z_bus` lays them, but PRESSED_PITCH
    apart, so that every one is at clearance to its neighbours."""
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
        for i, y in enumerate(i * PRESSED_PITCH for i in range(count))
    ]


@dataclass(frozen=True)
class JointCase:
    chains: Callable[[], list[ChainInput]]
    params: EnergyParams


JOINT_PARAMS = replace(scenarios.PARAMS, repulsion=BUS_REPULSION)


def joint_cases() -> dict[str, JointCase]:
    return {
        "lone alone": JointCase(lambda: [lone_track()], JOINT_PARAMS),
        "lone beside its copy": JointCase(
            lambda: [lone_track(), replace_net(lone_track(), "copy", LONE_OFFSET)],
            JOINT_PARAMS,
        ),
        "lone beside 10 Z tracks at 0.6 mm": JointCase(
            lambda: [lone_track(), *z_bus(0.0)], JOINT_PARAMS
        ),
        **{
            f"lone beside {n} pressed Z tracks": JointCase(
                lambda n=n: [lone_track(), *pressed_bus(n)], JOINT_PARAMS
            )
            for n in (3, 5, 10)
        },
    }


def replace_net(chain: ChainInput, net: str, shift: float) -> ChainInput:
    """`chain` as a chain of `net`, `shift` mm lower in y."""
    points = chain.centreline.points + np.array([0.0, shift])
    centreline = replace(chain.centreline, description=net, net=net, points=points)
    return replace(chain, centreline=centreline)


def joint_run(case: JointCase) -> tuple[Case, dict[str, ChainTrace]]:
    """The run of `case`, and the traces it fills as it runs."""
    settings = smooth_options().settings
    traces: dict[str, ChainTrace] = {}

    def go() -> tuple[int, str]:
        with traced() as trace:
            result = optimize_chains(case.chains(), case.params, settings)
        traces.update(trace)
        lone = trace[LONE]
        return result.iterations, (
            f"{result.termination.value}; lone {lone.lengths[-1]:.4f} mm "
            f"after {len(lone.lengths)} bursts, {lone.seconds[-1]:.2f} s; "
            f"all {sum(length(c.polyline) for c in result.chains):.4f} mm"
        )

    return go, traces


REACHED = smooth_options().settings.stall_tolerance
"""How near the alone run's final length the lone track counts as there, mm:
the runs' stall tolerance."""


def plot_traces(
    name: str, alone: ChainTrace, trace: dict[str, ChainTrace], directory: Path
) -> Path:
    """The lone track's length and shape change per outer iteration, alone
    and in the case, and the case's other chains' lengths, as a PNG."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 10))
    lone_lengths, moves, others = figure.subplots(3, 1, sharex=True)
    final = alone.lengths[-1]
    runs = (("alone", alone, "C0"), ("in the case", trace[LONE], "C3"))
    for label, t, colour in runs:
        rounds = range(1, len(t.lengths) + 1)
        lone_lengths.plot(
            rounds, [x - final for x in t.lengths], color=colour, label=label
        )
        moves.semilogy(rounds, t.moves, color=colour, label=label)
    lone_lengths.set_yscale("symlog", linthresh=REACHED)
    lone_lengths.set_ylabel("lone track above its alone length, mm")
    lone_lengths.legend()
    moves.set_ylabel("lone track's shape change, mm")
    for chain, t in trace.items():
        if chain != LONE:
            rounds = range(1, len(t.lengths) + 1)
            others.plot(rounds, [x - t.lengths[0] for x in t.lengths], linewidth=0.8)
    others.set_ylabel("other chains' length change, mm")
    others.set_xlabel("outer iteration")
    lone_lengths.set_title(name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name.replace(' ', '-')}.png"
    figure.savefig(path, dpi=100)  # pyright: ignore[reportUnknownMemberType]
    return path


def reached(trace: ChainTrace, target: float) -> str:
    """The first burst after which the track is within REACHED of `target`."""
    for k, x in enumerate(trace.lengths):
        if x <= target + REACHED:
            return f"burst {k + 1}, {trace.seconds[k]:.2f} s"
    return "never"


def joint() -> None:
    print(HEADER)
    alone: ChainTrace | None = None
    lines: list[str] = []
    for name, case in joint_cases().items():
        if alone is not None and ARGS.case not in name:
            continue
        run, trace = joint_run(case)
        measure(name, run)
        lone = trace[LONE]
        if alone is None:
            alone = lone
            continue
        settled = len(alone.lengths)
        at = min(settled, len(lone.lengths)) - 1
        lines.append(
            f"- {name}: lone {lone.lengths[at] - alone.lengths[at]:+.4f} mm against "
            f"alone at burst {at + 1} (alone settled at {settled}, "
            f"{alone.lengths[-1]:.4f} mm); within {REACHED:g} mm of it at "
            f"{reached(lone, alone.lengths[-1])}; alone at "
            f"{reached(alone, alone.lengths[-1])}"
        )
        if ARGS.plot is not None:
            lines.append(f"  plot: {plot_traces(name, alone, trace, ARGS.plot)}")
    print("\n".join(lines))


RETURN_CASES = {
    "unselected GND via 1.5 mm across the route": ((5.0, 1.5), False),
    "unselected GND via 1.5 mm diagonally": ((6.06, 1.06), False),
    "stitching GND via 2.5 mm across the route": ((5.0, 2.5), True),
}
RETURN_SETTINGS = {"defaults": [], "--repulsion 0": ["--repulsion", "0"]}
RETURN_HELPERS = ("return_via_board", "RETURN_VIAS_APART", "neutral_distance")
"""What the return section builds and measures with from the measured
tree's `tests/conftest.py`."""


def return_run(
    ground: tuple[float, float], stitching: bool, args: Sequence[str]
) -> tuple[float, float, EnergyParams]:
    """The run's summed chain length and the distance between the SIG and
    GND vias where it leaves them, both mm, and its energy parameters."""
    connect(conftest.return_via_board(ground, stitching=stitching))
    options = cli.smooth_options(cli.parse_args(["smooth", "--dry-run", *args]))
    reports: list[pipeline.OptimizerReport] = []
    pipeline.run_smooth(options, reports.append)
    (report,) = reports
    ends = {v.id: v.end for v in report.vias}
    signal = ends["v"]
    return_via = ends.get("g", ground)
    length = sum(c.after.length for c in report.chains)
    return length, math.dist(signal, return_via), options.params


def return_vias() -> None:
    """Where the SIG via and its return via settle, against the neutral
    distance their energy gives them, and how the chains' length compares
    with the run without attraction."""
    print(
        "| case | settings | distance mm | neutral distance mm "
        "| chains vs `--attraction 0`, µm |\n|---|---|---|---|---|"
    )
    for name, (ground, stitching) in RETURN_CASES.items():
        for label, args in RETURN_SETTINGS.items():
            off, _, _ = return_run(ground, stitching, [*args, "--attraction", "0"])
            length, distance, params = return_run(ground, stitching, args)
            neutral = (
                conftest.neutral_distance(params, conftest.RETURN_VIAS_APART)
                if params.repulsion
                else conftest.RETURN_VIAS_APART + params.standoff
            )
            print(
                f"| {name} | {label} | {distance:.3f} | {neutral:.3f} "
                f"| {1000 * (length - off):+.1f} |",
                flush=True,
            )


def main() -> None:
    logger.remove()
    print(f"Source: {TREE}\n")
    if ARGS.only in (None, "grid"):
        grid()
        print()
    if ARGS.only in (None, "crossed"):
        crossed()
        print()
    if ARGS.only in (None, "scenarios"):
        scenarios_table()
        print()
    if ARGS.only in (None, "bus"):
        bus()
        print()
    if ARGS.only in (None, "joint"):
        joint()
        print()
    if ARGS.only in (None, "return"):
        fields = EnergyParams.__dataclass_fields__
        missing = [
            *(() if "attraction" in fields else ("return-via attraction",)),
            *(f"conftest.{h}" for h in RETURN_HELPERS if not hasattr(conftest, h)),
        ]
        if missing:
            print(f"return: skipped, the tree has no {', '.join(missing)}")
        else:
            return_vias()


if __name__ == "__main__":
    main()
