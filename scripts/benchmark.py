"""Measure the optimizer: wall and CPU time, outer iterations, bursts,
energy evaluations and the farthest move in one burst on stitched ground
planes and on the synthetic scenarios, and where return vias settle.

Usage: ./run bench [--tree DIR] [--only grid|crossed|scenarios|return]
                   [--case TEXT]
                   [-- SMOOTH_ARGS]

`--tree` measures another checkout's source and tests, such as an older
commit unpacked with `git archive`, so that two versions can be compared
on one machine. `--case` runs only the planes and grids whose names
contain TEXT.
Arguments after `--` go to `kibends smooth` for the plane runs and give
the crossed grids' energy parameters and optimizer settings, a
`--time-limit` among them overriding TIME_LIMIT, so a run can be measured
at other settings. Each table row is one run.

- grid: two-layer GND planes whose stitching vias are all selected, among
  them a 30 mm plane with 194 vias at 1.72 mm pitch, with and without three
  signal routes that change layers through a via inside it, run end to end
  through a fake KiCad as `kibends smooth --dry-run` runs, setup included,
  each stopped after TIME_LIMIT seconds unless it converges first.
- crossed: GND stitching grids in a GND zone that signal routes cross,
  each through the optimizer at default settings, stopped after TIME_LIMIT
  seconds unless it converges first.
- scenarios: the cases of `tests/scenarios.py`, each through the optimizer
  at default settings.
- return: a signal via and a ground via settling together, at default
  settings and without repulsion; skipped for a tree without return-via
  attraction or without the test helpers it builds its boards with.
"""

import argparse
import math
import sys
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the optimizer.")
    parser.add_argument("--tree", type=Path, default=ROOT)
    parser.add_argument("--only", choices=("grid", "crossed", "scenarios", "return"))
    parser.add_argument("--case", default="", help="run only the cases naming this")
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
from kibends.optimize import optimize, optimize_chains  # noqa: E402
from kibends.snapshot import ViaZone  # noqa: E402
from kibends.vias import ViaInput  # noqa: E402

assert Path(kibends.__file__).is_relative_to(TREE), kibends.__file__


@dataclass
class Count:
    bursts: int = 0
    evaluations: int = 0
    step: float = 0.0
    """The farthest any vertex or via centre moved in one burst, mm."""


@contextmanager
def counted() -> Generator[Count]:
    """Count the bursts the optimizer runs and the energy evaluations in
    them."""
    count = Count()
    real = optimize_module.run_burst

    def burst(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
    ) -> Burst:
        count.bursts += 1

        def evaluated(x: FloatArray) -> tuple[float, FloatArray]:
            count.evaluations += 1
            return objective(x)

        result = real(evaluated, start, revert, settings)
        if len(start):
            moved = (result.points - start).reshape(-1, 2)
            count.step = max(count.step, float(np.linalg.norm(moved, axis=1).max()))
        return result

    setattr(optimize_module, "run_burst", burst)  # noqa: B010
    try:
        yield count
    finally:
        setattr(optimize_module, "run_burst", real)  # noqa: B010


@dataclass(frozen=True)
class Measured:
    wall: float
    cpu: float
    iterations: int
    count: Count
    note: str


def measure(name: str, work: Callable[[], tuple[int, str]]) -> Measured:
    with counted() as count:
        wall, cpu = time.perf_counter(), time.process_time()
        iterations, note = work()
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    measured = Measured(wall, cpu, iterations, count, note)
    print(
        f"| {name} | {wall:.2f} | {cpu:.2f} | {iterations} | {count.bursts} "
        f"| {count.evaluations} | {count.step:.2f} | {note} |",
        flush=True,
    )
    return measured


HEADER = (
    "| case | wall s | CPU s | iterations | bursts | evaluations | largest step mm "
    "| result |\n"
    "|---|---|---|---|---|---|---|---|"
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


def plane_run(plane: Plane) -> Callable[[], tuple[int, str]]:
    board = plane.board()
    options = replace(smooth_options(), dry_run=True)

    def go() -> tuple[int, str]:
        connect(board)
        reports: list[pipeline.OptimizerReport] = []
        pipeline.run_smooth(options, reports.append)
        (report,) = reports
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


def scenario_cases() -> dict[str, Case]:
    def single(make: Callable[[], scenarios.Scenario]) -> Case:
        def go() -> tuple[int, str]:
            s = make()
            result = optimize(s.polyline, s.fixed, s.field, s.params)
            outcome = result.termination.value
            return result.iterations, f"{outcome}, {length(result.polyline):.4f} mm"

        return go

    def several(make: Callable[[], Several]) -> Case:
        def go() -> tuple[int, str]:
            s = make()
            result = optimize_chains(
                s.chains,
                s.params,
                vias=s.vias if isinstance(s, ViasScenario) else (),
                pairs=s.pairs if isinstance(s, PairsScenario) else (),
            )
            total = sum(length(c.polyline) for c in result.chains) + sum(
                length(t) for p in result.pairs for t in p.kept()
            )
            return result.iterations, f"{result.termination.value}, {total:.4f} mm"

        return go

    free = FreeSpace()
    return {
        "wrap": single(scenarios.wrap),
        "channel-0": single(lambda: scenarios.channel(0.0)),
        "channel-1": single(lambda: scenarios.channel(1.0)),
        "topology": single(scenarios.topology),
        "pad-row": single(scenarios.pad_row),
        "frozen": single(scenarios.frozen),
        "detour": single(scenarios.detour),
        "narrow-gap": single(scenarios.narrow_gap),
        "channel-pair": several(scenarios.channel_pair),
        "uneven-channel-pair": several(scenarios.uneven_channel_pair),
        "tent": several(scenarios.tent),
        "same-net-wrap": several(scenarios.same_net_wrap),
        "crowded": several(scenarios.crowded),
        "drape": several(scenarios.drape),
        "squeeze": several(scenarios.squeeze),
        "hooks": several(scenarios.hooks),
        "neck-down": several(scenarios.neck_down),
        "centreline-over-pad": several(scenarios.centreline_over_pad),
        "via-detour": several(scenarios.via_detour),
        "via-on-one-layer": several(scenarios.via_on_one_layer),
        "via-fermat": several(scenarios.via_fermat),
        "via-hemmed": several(scenarios.via_hemmed),
        "via-corner": several(scenarios.via_corner),
        "via-beside-chain": several(scenarios.via_beside_chain),
        "via-pair": several(scenarios.via_pair),
        "via-over-pinned-N": several(lambda: scenarios.via_over_pinned("N")),
        "via-over-pinned-P": several(lambda: scenarios.via_over_pinned("P")),
        "via-over-bowed-N": several(lambda: scenarios.via_over_bowed("N")),
        "via-over-bowed-P": several(lambda: scenarios.via_over_bowed("P")),
        "via-detour-over": several(lambda: scenarios.via_detour_over(free)),
        "via-detour-over-box": several(
            lambda: scenarios.via_detour_over(Box(4.9, 2.0, 0.05, 0.05))
        ),
        "via-slack": several(lambda: scenarios.via_slack(free)),
        "via-across-chain": several(scenarios.via_across_chain),
        "via-grid-3x4": several(lambda: scenarios.via_grid(3, 4)),
        "pair-over-pad": several(scenarios.pair_over_pad),
        "pair-tent": several(scenarios.pair_tent),
        "pair-corner": several(scenarios.pair_corner),
        "pair-round-corner": several(scenarios.pair_round_corner),
        "pair-uneven-end": several(scenarios.pair_uneven_end),
        "pair-chamfered": several(scenarios.pair_chamfered),
        "pair-split": several(scenarios.pair_split),
        "pair-turning-back-150": several(scenarios.pair_turning_back),
    }


def length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def scenarios_table() -> None:
    print(HEADER)
    total = Count()
    wall = cpu = 0.0
    iterations = 0
    for name, case in scenario_cases().items():
        m = measure(name, case)
        wall, cpu, iterations = wall + m.wall, cpu + m.cpu, iterations + m.iterations
        total.bursts += m.count.bursts
        total.evaluations += m.count.evaluations
        total.step = max(total.step, m.count.step)
    print(
        f"| **all {len(scenario_cases())}** | {wall:.2f} | {cpu:.2f} | {iterations} "
        f"| {total.bursts} | {total.evaluations} | {total.step:.2f} | |"
    )


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
