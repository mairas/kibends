"""The optimizer gives the same bits whatever the size of the rayon pool its
evaluations spread over (`kibends_core::parallel`)."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import FakeBoard, in_pool, kipy_pad, layer_snapshot
from distance_fields import FloatArray, FreeSpace
from scenarios import ChainsScenario, PairsScenario, ViasScenario, selected

import kibends.optimize as optimize_module
from kibends.bursts import (
    DEFAULT_SETTINGS,
    Burst,
    Check,
    OptimizerSettings,
    ScaledObjective,
    Stop,
)
from kibends.obstacles import build_obstacles
from kibends.optimize import optimize_chains
from kibends.results import RunResult

MANY = 8
"""Threads of the larger pool."""

SHORT = replace(DEFAULT_SETTINGS, max_iterations=3)
"""Enough outer iterations for every scenario to search its lists again."""

BUS_TRACKS = 4
PITCH = 0.6


def z_bus() -> ChainsScenario:
    """Nested Z tracks of as many nets in free space, their points many
    chunks."""
    return ChainsScenario(
        "z-bus",
        tuple(
            selected(
                f"S{i}",
                f"S{i}",
                FreeSpace(),
                (0.0, i * PITCH),
                (5.0 - i * PITCH, i * PITCH),
                (8.0 - i * PITCH, i * PITCH + 3.0),
                (12.0, i * PITCH + 3.0),
            )
            for i in range(BUS_TRACKS)
        ),
    )


def pad_rows() -> ChainsScenario:
    """Tracks between rows of pads, each pad of its own net, so that the
    static fields answer from their edges."""
    pads = [
        kipy_pad(f"p{row}_{k}", f"P{row}_{k}", k * 1.0, (row + 0.5) * 1.0)
        for row in range(-1, 3)
        for k in range(11)
    ]
    snap = layer_snapshot(FakeBoard(pads=pads), "F.Cu")
    return ChainsScenario(
        "pad-rows",
        tuple(
            selected(
                f"S{i}",
                f"S{i}",
                build_obstacles(snap, f"S{i}", scenarios.PAIR_WIDTH),
                (0.0, float(i)),
                (5.0, i + 0.1),
                (10.0, float(i)),
            )
            for i in range(3)
        ),
    )


Scenario = ChainsScenario | ViasScenario | PairsScenario

SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "chains": scenarios.crowded,
    "pairs": scenarios.pair_tent,
    "vias": scenarios.via_pair,
    "bus": z_bus,
    "pads": pad_rows,
}


def run(s: Scenario) -> RunResult:
    return optimize_chains(
        s.chains,
        s.params,
        SHORT,
        vias=s.vias if isinstance(s, ViasScenario) else (),
        pairs=s.pairs if isinstance(s, PairsScenario) else (),
    )


def bits(result: RunResult) -> list[bytes]:
    """Every coordinate the run gives, as bytes."""
    arrays = [c.polyline for c in result.chains]
    arrays += [a for p in result.pairs for a in (p.centreline, p.a, p.b)]
    arrays += [np.array(v.position) for v in result.vias]
    return [np.ascontiguousarray(a, dtype=np.float64).tobytes() for a in arrays]


@pytest.mark.parametrize("make", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_a_run_gives_the_same_bits_on_any_thread_count(
    make: Callable[[], Scenario],
) -> None:
    one = in_pool(1, lambda: run(make()))
    many = in_pool(MANY, lambda: run(make()))

    assert (many.termination, many.iterations) == (one.termination, one.iterations)
    assert bits(many) == bits(one)


@pytest.mark.parametrize("make", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_each_evaluation_gives_the_same_bits_on_any_thread_count(
    make: Callable[[], Scenario], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = optimize_module.run_burst
    compared: list[bool] = []

    def checked(
        objective: ScaledObjective,
        start: FloatArray,
        revert: Check,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        core = objective.objective
        one = in_pool(1, lambda: core(start))
        many = in_pool(MANY, lambda: core(start))
        compared.append(
            np.float64(one[0]).tobytes() == np.float64(many[0]).tobytes()
            and one[1].tobytes() == many[1].tobytes()
        )
        return real(objective, start, revert, settings, stop)

    monkeypatch.setattr(optimize_module, "run_burst", checked)
    run(make())

    assert compared and all(compared)
